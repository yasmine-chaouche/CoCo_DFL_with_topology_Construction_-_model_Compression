"""CoCo: joint topology Construction and model Compression for decentralized
federated learning (Wang et al., IEEE TMC 2023) -- simulation entry point.

Examples
  python main.py --algo coco  --dataset cifar10  --model vgg9
  python main.py --algo choco --ratio 0.5 --dataset cifar10 --model vgg9
  python main.py --algo coco  --dataset cifar100 --model resnet9 --non-iid 20
  python main.py --algo coco  --dataset unsw-nb15 --partition dirichlet --alpha 0.5
"""
import argparse

from coco.simulator import Simulator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="coco",
                   choices=["coco", "dcd", "choco", "saps", "lct", "fcr", "fnt-ring", "fnt-fc"])
    p.add_argument("--ratio", type=float, default=0.5,
                   help="fixed compression ratio for dcd/choco/saps/fcr")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "unsw-nb15"])
    p.add_argument("--model", default=None, choices=["vgg9", "resnet9", "smallcnn", "mlp"],
                   help="default: vgg9 (cifar10), resnet9 (cifar100), mlp (unsw-nb15)")
    p.add_argument("--unsw-task", default="multiclass", choices=["multiclass", "binary"],
                   help="unsw-nb15: 10 attack categories or normal/attack")
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--partition", default="auto", choices=["auto", "iid", "dominant", "missing", "dirichlet"],
                   help="auto: dominant (cifar10), missing (cifar100), iid (unsw-nb15)")
    p.add_argument("--non-iid", type=float, default=None,
                   help="dominant: %% of samples from a dominant class (100/C = IID); "
                        "missing: number of missing classes (0 = IID)")
    p.add_argument("--alpha", type=float, default=0.5, help="dirichlet concentration (smaller = more skewed)")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--epochs", type=float, default=None,
                   help="default: 200 (cifar10) / 300 (cifar100) / 30 (unsw-nb15)")
    p.add_argument("--rounds", type=int, default=200, help="overrides --epochs")
    p.add_argument("--tau", type=int, default=50, help="local updates per round")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: 0.05 (vgg9) / 0.1 (resnet9)")
    p.add_argument("--lr-decay", type=float, default=None, help="per-epoch decay, default 0.98 / 0.99")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # CoCo
    p.add_argument("--r-min", type=float, default=0.01, help="smallest compression ratio")
    p.add_argument("--hops", type=int, default=1,
                   help="gossip steps per round (2 = models travel two hops per round; "
                        "communication time and traffic are multiplied accordingly)")
    p.add_argument("--r-max", type=float, default=1.0,
                   help="largest compression ratio CoCo may choose (1 = no cap)")
    p.add_argument("--beta1", type=float, default=0.5, help="EMA for estimated consensus distance (Eq. 26)")
    p.add_argument("--beta2", type=float, default=0.1, help="EMA for D_max (Eq. 28)")
    p.add_argument("--dmax-scale", type=float, default=1.0, help="multiplier applied to D_max")
    p.add_argument("--tiebreak", default="min_traffic", choices=["min_traffic", "max_ratio"],
                   help="which optimal LP solution to use")
    p.add_argument("--compressor", default="randomk", choices=["randomk", "topk"])
    # network
    p.add_argument("--network", default="heter_dynamic",
                   choices=["heter_dynamic", "heter_static", "homo_static"])
    p.add_argument("--volatility", type=float, default=0.3, help="bandwidth random-walk std (dynamic)")
    p.add_argument("--comp-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="computation time of tau local steps (s); default depends on the model")
    p.add_argument("--swap-at", type=float, default=None,
                   help="epoch at which workers 0 and m-1 exchange bandwidth (Fig. 13b)")
    # misc
    p.add_argument("--eval-every", type=int, default=None,
                   help="evaluate every N rounds (default: at the end of every epoch, as in the paper)")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--track-workers", action="store_true", help="log per-worker traffic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    ds = a.dataset
    a.model = a.model or {"cifar10": "vgg9", "cifar100": "resnet9", "unsw-nb15": "mlp"}[ds]
    if a.partition == "auto":
        a.partition = {"cifar10": "dominant", "cifar100": "missing", "unsw-nb15": "iid"}[ds]
    if a.non_iid is None:
        a.non_iid = {"dominant": 10, "missing": 0}.get(a.partition, 0)
    a.epochs = a.epochs or {"cifar10": 200, "cifar100": 300, "unsw-nb15": 30}[ds]
    a.lr = a.lr or (0.05 if a.model != "resnet9" else 0.1)
    a.lr_decay = a.lr_decay or (0.98 if a.model != "resnet9" else 0.99)
    if a.name is None:
        tag = a.algo if a.algo in ("coco", "lct", "fnt-ring", "fnt-fc") else f"{a.algo}{a.ratio}"
        if a.algo == "coco" and a.r_max < 1:
            tag += f"-rmax{a.r_max:g}"
        if a.algo == "coco" and a.dmax_scale != 1:
            tag += f"-dmax{a.dmax_scale:g}"
        split = {"iid": "iid", "dirichlet": f"dir{a.alpha:g}"}.get(a.partition, f"niid{a.non_iid:g}")
        task = f"-{a.unsw_task}" if ds == "unsw-nb15" else ""
        if a.hops > 1:
            tag += f"-{a.hops}hop"
        a.name = f"{ds}{task}_{a.model}_{tag}_{split}_{a.network}_s{a.seed}"
    return a


if __name__ == "__main__":
    Simulator(parse_args()).run()
