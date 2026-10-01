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

COMP_RANGE = {"vgg9": (2.5, 7.5), "resnet9": (7.5, 12.5), "smallcnn": (2.5, 7.5)}


class Simulator:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.device = torch.device(args.device)
        m = self.m = args.workers

        self.data = GPUData(args.dataset, args.data_dir, m, args.non_iid, args.seed, self.device)
        base = build_model(args.model, self.data.num_classes).to(self.device)
        self.models = [build_model(args.model, self.data.num_classes).to(self.device) for _ in range(m)]
        for mod in self.models:  # identical initialisation
            mod.load_state_dict(base.state_dict())
        self.opts = [torch.optim.SGD(mod.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay) for mod in self.models]
        self.dim = get_flat(base).numel()
        self.model_size = self.dim * 32 / 1e6  # Mb (float32)

        self.net = NetworkSim(m, args.network, self.rng, comp_range=COMP_RANGE[args.model],
                              volatility=args.volatility)
        self.planner = build_planner(args.algo, m, args.ratio, args.r_min, args.tiebreak)
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
        workers = range(self.m) if workers is None else workers
        accs = []
        for i in workers:
            model = self.models[i]
            model.eval()
            correct = 0
            for s in range(0, len(self.data.y_test), 2000):
                x, y = self.data.x_test[s:s + 2000], self.data.y_test[s:s + 2000]
                correct += (model(x).argmax(1) == y).sum().item()
            accs.append(correct / len(self.data.y_test))
        return float(np.mean(accs))

    # ------------------------------------------------------------------ round
    def run(self):
        a = self.args
        rounds_per_epoch = self.data.samples_per_worker / (a.tau * a.batch_size)
        total_rounds = a.rounds or math.ceil(a.epochs * rounds_per_epoch)
        print(f"model={a.model} d={self.dim} (B={self.model_size:.1f} Mb) workers={self.m} "
              f"algo={a.algo} rounds={total_rounds} ({rounds_per_epoch:.2f} rounds/epoch)")

        for k in range(total_rounds):
            epoch = k / rounds_per_epoch
            lr = a.lr * a.lr_decay ** int(epoch)
            if a.swap_at is not None and epoch >= a.swap_at and not self.swapped:
                self.net.swap(0, self.m - 1)  # Fig. 13b
                self.swapped = True
            b_in, b_out, t_cp = self.net.step()

            # ---- coordinator: Algorithm 2, CoCo (Algorithm 3) or baseline
            t0 = time.time()
            ctx = PlanContext(self.A_base, self.estimator.D, self.dmax.get(), b_in, b_out,
                              t_cp, self.model_size, k)
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

            # ---- element-wise aggregation (Eqs. 6-7)
            w = 1.0 / (A.sum(axis=1).max() + 1)
            for i in range(self.m):
                new = X[i].clone()
                for j in np.nonzero(A[i])[0]:
                    new += w * (X[j] - X[i]) * Z[j]
                set_flat(self.models[i], new)

            # ---- bookkeeping: time (Eq. 18), traffic, statuses for the next round
            t_round = round_time(A, r, b_in, b_out, t_cp, self.model_size)
            self.sim_time += t_round
            self.traffic += float((self.model_size / 8) * (r * A.sum(axis=1)).sum())
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
            }
            if a.track_workers:
                rec["worker_traffic_MB"] = ((self.model_size / 8) * r * A.sum(axis=1)).tolist()
            last = k == total_rounds - 1
            if (k + 1) % a.eval_every == 0 or last:
                rec["accuracy"] = self.evaluate()
            self.history.append(rec)
            if "accuracy" in rec or (k + 1) % a.log_every == 0:
                print(f"[r{k + 1:4d} ep{rec['epoch']:6.1f}] t={self.sim_time:9.0f}s "
                      f"traffic={self.traffic:9.0f}MB loss={rec['loss']:.3f} "
                      f"edges={rec['edges']:3d} r={rec['ratio_mean']:.3f} "
                      f"[{rec['ratio_min']:.2f},{rec['ratio_max']:.2f}] "
                      f"CD true={true_cd:.3f} est={est_before:.3f} Dmax={dmax:.3f}"
                      + (f" acc={rec['accuracy']:.4f}" if "accuracy" in rec else ""), flush=True)
        self.save()
        return self.history

    def save(self):
        os.makedirs(self.args.out_dir, exist_ok=True)
        path = os.path.join(self.args.out_dir, f"{self.args.name}.json")
        with open(path, "w") as f:
            json.dump({"config": vars(self.args), "history": self.history}, f)
        print(f"saved {path}")
