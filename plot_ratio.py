"""Accuracy, macro-F1, consensus distance, time and traffic as a function of the
compression ratio.

  python plot_ratio.py results/ratio_sweep/*.json -o figures/ratio_sweep/ratio_sweep.png

Writes <output> and <output stem>_consensus.png (consensus distance over training,
one curve per run, coloured by ratio).

One curve per (algorithm, data partition, network) group of fixed-ratio runs
(dcd / choco / saps / fcr); x = the run's --ratio. CoCo runs, whose ratios are
adaptive, are placed at their average ratio over training: a single run is a star,
several runs (e.g. from sweep_coco.sh, varying --r-max or --dmax-scale) are joined
into a curve and labelled with their setting. Keep r_max and D_max sweeps in
separate folders / plots.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIXED = ("dcd", "choco", "saps", "fcr")
NAMES = {"dcd": "DCD (degree 5)", "choco": "Choco (ring)", "saps": "SAPS", "fcr": "FCR",
         "coco": "CoCo (adaptive)"}


def summarize(path, last, cd_frac, cd_key):
    with open(path) as f:
        run = json.load(f)
    cfg, hist = run["config"], run["history"]
    evals = [h for h in hist if "accuracy" in h]
    tail = evals[-last:]
    cd = np.array([h[cd_key] for h in hist])
    n_tail = max(1, int(round(cd_frac * len(cd))))
    split = {"iid": "iid", "dirichlet": f"dir{cfg.get('alpha', 0):g}"}.get(
        cfg.get("partition"), f"niid{cfg.get('non_iid', 0):g}")
    return {
        "algo": cfg["algo"],
        "group": f"{cfg['dataset']} {split} {cfg['network']}",
        "ratio": cfg["ratio"] if cfg["algo"] in FIXED else float(np.mean([h["ratio_mean"] for h in hist])),
        "accuracy": float(np.mean([h["accuracy"] for h in tail])),
        "f1_macro": float(np.mean([h.get("f1_macro", np.nan) for h in tail])),
        "best_accuracy": max(h["accuracy"] for h in evals),
        "time": hist[-1]["time"],
        "traffic": hist[-1]["traffic_MB"],
        "epochs": hist[-1]["epoch"],
        "cd_final": float(cd[-n_tail:].mean()),
        "cd_mean": float(cd.mean()),
        "cd_curve": (np.array([h["epoch"] for h in hist]), cd),
        "file": os.path.basename(path),
        "setting": (f"rmax={cfg.get('r_max', 1):g}" if cfg.get("r_max", 1) < 1 else
                    f"Dmax×{cfg.get('dmax_scale', 1):g}" if cfg.get("dmax_scale", 1) != 1 else "default"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--last", type=int, default=3,
                   help="average the last N evaluations (reduces noise)")
    p.add_argument("--linear-x", action="store_true", help="linear instead of log ratio axis")
    p.add_argument("--cd-frac", type=float, default=0.1,
                   help="final consensus distance = mean over the last fraction of rounds")
    p.add_argument("--cd", default="post", choices=["post", "pre"],
                   help="consensus distance after aggregation (post) or before it (pre)")
    p.add_argument("--smooth", type=int, default=5, help="moving average (rounds) in the consensus figure")
    p.add_argument("-o", "--output", default="ratio_sweep.png")
    a = p.parse_args()
    # a bare file name goes to figures/, never to the project root
    if not os.path.dirname(a.output):
        a.output = os.path.join("figures", a.output)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    missing = [f for f in a.files if not os.path.isfile(f)]
    for f in missing:
        print(f"warning: no such file: {f}"
              + ("  (the pattern matched nothing: has that sweep been run yet?)" if "*" in f else ""))
    files = [f for f in a.files if os.path.isfile(f)]
    if not files:
        raise SystemExit("no result files to plot")
    cd_key = "consensus_true" if a.cd == "post" else "consensus_true_pre"
    runs = [summarize(f, a.last, a.cd_frac, cd_key) for f in files]
    curves = defaultdict(list)
    for r in runs:
        curves[(r["algo"], r["group"])].append(r)

    panels = [("accuracy", f"test accuracy "),
              ("f1_macro", f"macro-F1 "),
              ("cd_final", f"consensus distance (last {a.cd_frac:.0%} of rounds)"),
              ("time", "total simulated time (s)"),
              ("traffic", "total traffic (MB)"),
              ("cd_mean", "consensus distance (mean over training)")]
    fig, grid = plt.subplots(2, 3, figsize=(18, 9))
    axes = grid.ravel()
    groups = sorted({g for _, g in curves})
    multi = len(groups) > 1
    print(f"{'run':62s} {'ratio':>6s} {'acc':>7s} {'f1':>7s} {'CD end':>7s} {'CD mean':>7s} "
          f"{'time(s)':>9s} {'traffic(MB)':>11s}")
    coco_colors = ["crimson", "darkorange", "purple", "saddlebrown", "black"]
    for (algo, group), rs in sorted(curves.items()):
        rs.sort(key=lambda r: r["ratio"])
        star = coco_colors[groups.index(group) % len(coco_colors)]
        label = NAMES.get(algo, algo) + (f" [{group}]" if multi else "")
        x = [r["ratio"] for r in rs]
        for ax, (key, _) in zip(axes, panels):
            y = [r[key] for r in rs]
            if algo in FIXED:
                ax.plot(x, y, marker="o", label=label)
            else:
                ax.plot(x, y, marker="*", markersize=14, zorder=5, color=star,
                        linestyle="-" if len(rs) > 1 else "none", label=label)
                if len(rs) > 1:
                    for r, xi, yi in zip(rs, x, y):
                        ax.annotate(r["setting"], (xi, yi), textcoords="offset points",
                                    xytext=(4, 6), fontsize=7, color=star)
        for r in rs:
            print(f"{r['file'][:62]:62s} {r['ratio']:6.3f} {r['accuracy']:7.4f} {r['f1_macro']:7.4f} "
                  f"{r['cd_final']:7.3f} {r['cd_mean']:7.3f} {r['time']:9.0f} {r['traffic']:11.0f}")
    for ax, (key, ylabel) in zip(axes, panels):
        ax.set_xlabel("compression ratio r" + (" (CoCo: average)" if any(k[0] == "coco" for k in curves) else ""))
        ax.set_ylabel(ylabel)
        if not a.linear_x:
            ax.set_xscale("log")
        if key in ("time", "traffic"):
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    epochs = sorted({round(r["epochs"]) for r in runs})
    fig.suptitle(f"Impact of the compression ratio ({', '.join(groups)}; {'/'.join(map(str, epochs))} epochs)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.output, dpi=120)
    print(f"saved {a.output}")
    plot_consensus_curves(runs, curves, a)


def smooth(v, w):
    if w <= 1:
        return v
    c = np.cumsum(np.insert(np.asarray(v, float), 0, 0.0))
    idx = np.arange(1, len(v) + 1)
    n = np.minimum(idx, w)
    return (c[idx] - c[idx - n]) / n


def plot_consensus_curves(runs, curves, a):
    """Consensus distance vs epochs: one panel per data split; fixed-ratio runs coloured by
    ratio (solid, one line style per algorithm), CoCo runs dashed in black."""
    groups = sorted({g for _, g in curves})
    ncol = min(2, len(groups))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 4.8 * nrow), squeeze=False)
    fixed = [r["ratio"] for r in runs if r["algo"] in FIXED] or [r["ratio"] for r in runs]
    norm = matplotlib.colors.LogNorm(vmin=min(fixed), vmax=max(fixed)) if not a.linear_x and min(fixed) > 0 \
        else matplotlib.colors.Normalize(vmin=min(fixed), vmax=max(fixed))
    cmap = plt.get_cmap("viridis")
    styles = dict(zip(FIXED, ["-", "-.", ":", "-"]))
    n_fixed_algos = len({k[0] for k in curves if k[0] in FIXED})
    for ax, group in zip(axes.ravel(), groups):
        for (algo, g), rs in sorted(curves.items()):
            if g != group:
                continue
            for i, r in enumerate(sorted(rs, key=lambda r: r["ratio"])):
                ep, cd = r["cd_curve"]
                if algo in FIXED:
                    tag = (f"{NAMES.get(algo, algo)} " if n_fixed_algos > 1 else "") + f"r={r['ratio']:.3g}"
                    ax.plot(ep, smooth(cd, a.smooth), color=cmap(norm(r["ratio"])),
                            linestyle=styles.get(algo, "-"), label=tag)
                else:
                    tag = f"CoCo (avg r={r['ratio']:.2f}" + (f", {r['setting']}" if r["setting"] != "default" else "") + ")"
                    ax.plot(ep, smooth(cd, a.smooth), color="black", linestyle="--", linewidth=1.8,
                            alpha=1 - 0.25 * i, label=tag)
        ax.set_title(group, fontsize=10)
        ax.set_xlabel("epochs")
        ax.set_ylabel("consensus distance" + (f" (moving avg, {a.smooth} rounds)" if a.smooth > 1 else ""))
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                 label="compression ratio", pad=0.01)
    stem, ext = os.path.splitext(a.output)
    path = f"{stem}_consensus{ext or '.png'}"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
