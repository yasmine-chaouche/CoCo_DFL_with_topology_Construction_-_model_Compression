"""Synchronous DFL simulation: workers (Algorithm 1) + coordinator (Algorithm 2).

All workers live in one process; wall-clock time of the edge system is simulated
from the bandwidth / computation model of Section 4.1.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .compression import COMPRESSORS
from .consensus import ConsensusEstimator, DmaxTracker, measured_distance, true_consensus_distance
from .data import GPUData
from .models import build_model, get_flat, set_flat
from .network import NetworkSim, round_time
from .planner import PlanContext, build_planner, complete

# tau * c_i (s): measured ranges from the paper for VGG9 / ResNet9; the MLP value is an assumption
COMP_RANGE = {"vgg9": (2.5, 7.5), "resnet9": (7.5, 12.5), "smallcnn": (2.5, 7.5), "mlp": (0.5, 1.5)}


class Simulator:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.device = torch.device(args.device)
        m = self.m = args.workers

        level = args.alpha if args.partition == "dirichlet" else args.non_iid
        self.data = GPUData(args.dataset, args.data_dir, m, args.partition, level, args.seed,
                            self.device, unsw_task=args.unsw_task)
        mk = lambda: build_model(args.model, self.data.num_classes, self.data.input_dim).to(self.device)
        base = mk()
        self.models = [mk() for _ in range(m)]
        for mod in self.models:  # identical initialisation
            mod.load_state_dict(base.state_dict())
        self.opts = [torch.optim.SGD(mod.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay) for mod in self.models]
        self.dim = get_flat(base).numel()
        self.model_size = self.dim * 32 / 1e6  # Mb (float32)

        comp_range = tuple(args.comp_range) if args.comp_range else COMP_RANGE[args.model]
        self.net = NetworkSim(m, args.network, self.rng, comp_range=comp_range,
                              volatility=args.volatility)
        self.planner = build_planner(args.algo, m, args.ratio, args.r_min, args.tiebreak, args.r_max)
        self.estimator = ConsensusEstimator(m, args.beta1)
        self.dmax = DmaxTracker(args.beta2, args.dmax_scale)
        self.compress = COMPRESSORS[args.compressor]
        self.A_base = complete(m)

        self.sim_time = 0.0
        self.traffic = 0.0  # MB
        self.history = []
        self.swapped = False

    # ----------------------------------------------------------------- worker
    def local_update(self, i, lr):
        """Algorithm 1 line 5 / Eq. (5): tau steps of local SGD. Returns ||g_i||."""
        model, opt = self.models[i], self.opts[i]
        for g in opt.param_groups:
            g["lr"] = lr
        before = get_flat(model)
        model.train()
        for _ in range(self.args.tau):
            x, y = self.data.batch(i, self.args.batch_size)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return float((get_flat(model) - before).norm().item()), float(loss.item())

    @torch.no_grad()
    def evaluate(self, workers=None):
        """Test accuracy and macro-F1, averaged over the workers' local models."""
        workers = range(self.m) if workers is None else workers
        C = self.data.num_classes
        accs, f1s = [], []
        for i in workers:
            model = self.models[i]
            model.eval()
            conf = torch.zeros(C * C, dtype=torch.long, device=self.device)
            for s in range(0, len(self.data.y_test), 4096):
                x, y = self.data.x_test[s:s + 4096], self.data.y_test[s:s + 4096]
                conf += torch.bincount(y * C + model(x).argmax(1), minlength=C * C)
            conf = conf.view(C, C).double()  # rows = true class, cols = prediction
            tp = conf.diag()
            prec = tp / conf.sum(0).clamp(min=1)
            rec = tp / conf.sum(1).clamp(min=1)
            f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
            present = conf.sum(1) > 0
            accs.append((tp.sum() / conf.sum()).item())
            f1s.append(f1[present].mean().item())
        return float(np.mean(accs)), float(np.mean(f1s))

    # ------------------------------------------------------------------ round
    def run(self):
        a = self.args
        rounds_per_epoch = self.data.samples_per_worker / (a.tau * a.batch_size)
        total_rounds = a.rounds or math.ceil(a.epochs * rounds_per_epoch)
        print(f"dataset={a.dataset} model={a.model} d={self.dim} (B={self.model_size:.2f} Mb) "
              f"workers={self.m} algo={a.algo} partition={a.partition} rounds={total_rounds} "
              f"({rounds_per_epoch:.2f} rounds/epoch)")
        counts = self.data.class_counts()
        print(f"local data: sizes {counts.sum(1).min()}-{counts.sum(1).max()}, "
              f"classes/worker {(counts > 0).sum(1).min()}-{(counts > 0).sum(1).max()} "
              f"of {self.data.num_classes}")

        for k in range(total_rounds):
            epoch = k / rounds_per_epoch
            lr = a.lr * a.lr_decay ** int(epoch)
            if a.swap_at is not None and epoch >= a.swap_at and not self.swapped:
                self.net.swap(0, self.m - 1)  # Fig. 13b
                self.swapped = True
            b_in, b_out, t_cp = self.net.step()

            # ---- coordinator: Algorithm 2, CoCo (Algorithm 3) or baseline
            t0 = time.time()
            # with H gossip steps per round the communication time is H * r * B / b, which is
            # the single-step formula with model size H * B: the planner optimises that time
            hops = a.hops
            ctx = PlanContext(self.A_base, self.estimator.D, self.dmax.get(), b_in, b_out,
                              t_cp, hops * self.model_size, k)
            A, r = self.planner.plan(ctx)
            plan_wall = time.time() - t0

            # ---- workers: local updating (Eq. 5)
            norms, losses = zip(*[self.local_update(i, lr) for i in range(self.m)])
            X = torch.stack([get_flat(mod) for mod in self.models])  # x_i^{(k,tau)}
            true_cd_pre = true_consensus_distance(X)

            # ---- compression (RandomK / TopK masks z_j)
            Z = torch.stack([self.compress(X[j], r[j], seed=a.seed * 100003 + k * 1009 + j)
                             for j in range(self.m)])

            # ---- exchange + consensus distance measurement (Eq. 23)
            reported = np.zeros((self.m, self.m))
            for i in range(self.m):
                for j in np.nonzero(A[i])[0]:
                    reported[i, j] = measured_distance(X[i], X[j], Z[j])

            # ---- element-wise aggregation (Eqs. 6-7), repeated for each gossip step ("hop"):
            # step h > 1 re-compresses the freshly averaged models (new RandomK masks) and
            # averages again with the same neighbours, so information travels h hops per round
            w = 1.0 / (A.sum(axis=1).max() + 1)
            for h in range(hops):
                if h > 0:
                    Z = torch.stack([self.compress(X[j], r[j], seed=a.seed * 100003 + k * 1009 + j
                                                   + h * 7919) for j in range(self.m)])
                X_next = X.clone()
                for i in range(self.m):
                    for j in np.nonzero(A[i])[0]:
                        X_next[i] += w * (X[j] - X[i]) * Z[j]
                X = X_next
            for i in range(self.m):
                set_flat(self.models[i], X[i])

            # ---- bookkeeping: time (Eq. 18), traffic, statuses for the next round
            t_round = round_time(A, r, b_in, b_out, t_cp, hops * self.model_size)
            self.sim_time += t_round
            self.traffic += float(hops * (self.model_size / 8) * (r * A.sum(axis=1)).sum())
            est_before = self.estimator.update(A, reported).sum() / self.m ** 2
            dmax = self.dmax.update(norms)
            with torch.no_grad():
                X_new = torch.stack([get_flat(mod) for mod in self.models])
            true_cd = true_consensus_distance(X_new)
            del X, Z, X_new

            rec = {
                "round": k + 1, "epoch": (k + 1) / rounds_per_epoch, "time": self.sim_time,
                "round_time": t_round, "traffic_MB": self.traffic, "lr": lr,
                "loss": float(np.mean(losses)), "edges": int(A.sum() // 2),
                "max_degree": int(A.sum(axis=1).max()), "ratio_mean": float(r.mean()),
                "ratio_min": float(r.min()), "ratio_max": float(r.max()),
                "consensus_true": true_cd, "consensus_true_pre": true_cd_pre,
                "consensus_est": float(est_before), "dmax": dmax,
                "update_norm": float(np.mean(norms)), "plan_wall_s": plan_wall,
                "ratios": [round(float(x), 4) for x in r],  # per-worker compression ratio
            }
            if a.track_workers:
                rec["worker_traffic_MB"] = (hops * (self.model_size / 8) * r * A.sum(axis=1)).tolist()
            last = k == total_rounds - 1
            if a.eval_every:  # every N rounds
                do_eval = (k + 1) % a.eval_every == 0
            else:  # end of each epoch, as in the paper: first round past an epoch boundary
                do_eval = int((k + 1) / rounds_per_epoch) > int(k / rounds_per_epoch)
            if do_eval or last:
                rec["accuracy"], rec["f1_macro"] = self.evaluate()
            self.history.append(rec)
            if "accuracy" in rec or (k + 1) % a.log_every == 0:
                print(f"[r{k + 1:4d} ep{rec['epoch']:6.1f}] t={self.sim_time:9.0f}s "
                      f"traffic={self.traffic:9.0f}MB loss={rec['loss']:.3f} "
                      f"edges={rec['edges']:3d} r={rec['ratio_mean']:.3f} "
                      f"[{rec['ratio_min']:.2f},{rec['ratio_max']:.2f}] "
                      f"CD true={true_cd:.3f} est={est_before:.3f} Dmax={dmax:.3f}"
                      + (f" acc={rec['accuracy']:.4f} f1={rec['f1_macro']:.4f}"
                         if "accuracy" in rec else ""), flush=True)
        self.save()
        return self.history

    def save(self):
        os.makedirs(self.args.out_dir, exist_ok=True)
        path = os.path.join(self.args.out_dir, f"{self.args.name}.json")
        with open(path, "w") as f:
            json.dump({"config": vars(self.args), "history": self.history,
                       "class_names": list(self.data.class_names),
                       "class_counts": self.data.class_counts().tolist()}, f)
        print(f"saved {path}")
