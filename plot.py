"""Plot accuracy (or macro-F1), training loss, consensus distance and compression ratios
for result files.

  python plot.py results/cifar10/01_coco_smoke_test/*.json --target 0.87 -o fig_cifar10.png

Writes <output> (3x3 grid) and, for runs that logged per-worker ratios,
<output stem>_ratios.png (heatmap worker x round of the compression ratio).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402


def load(path):
    with open(path) as f:
        run = json.load(f)
    hist = [h for h in run["history"] if "accuracy" in h]
    return run, hist


def short_labels(paths):
    """File names without the parts shared by every run (e.g. dataset, model, seed)."""
    names = [os.path.basename(p).replace(".json", "") for p in paths]
    if len(names) < 2:
        return names
    parts = [n.split("_") for n in names]
    pre = 0
    while all(len(p) > pre + 1 for p in parts) and len({p[pre] for p in parts}) == 1:
        pre += 1
    suf = 0
    while all(len(p) > pre + suf + 1 for p in parts) and len({p[-1 - suf] for p in parts}) == 1:
        suf += 1
    return ["_".join(p[pre:len(p) - suf]) for p in parts]


def smooth(values, window):
    """Trailing moving average (the per-round loss is noisy)."""
    if window <= 1:
        return values
    v = np.asarray(values, dtype=float)
    c = np.cumsum(np.insert(v, 0, 0.0))
    n = np.minimum(np.arange(1, len(v) + 1), window)
    return (c[1:] - c[np.arange(1, len(v) + 1) - n]) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--metric", default="accuracy", choices=["accuracy", "f1_macro"])
    p.add_argument("--target", type=float, default=None, help="target value of --metric for the summary table")
    p.add_argument("--smooth", type=int, default=5,
                   help="moving-average window (rounds) for the loss and the compression ratio")
    p.add_argument("--log-loss", action="store_true", help="log scale for the loss axis")
    p.add_argument("-o", "--output", default="fig_grid.png",
                   help="a bare file name is saved next to the first input .json")
    a = p.parse_args()
    # a bare file name goes next to the result files, so figures and .json stay together
    if not os.path.dirname(a.output):
        a.output = os.path.join(os.path.dirname(a.files[0]) or ".", a.output)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    fig, grid = plt.subplots(3, 3, figsize=(18, 12.6))
    # 0-2: metric vs epochs/time/traffic, 3-4: loss vs epochs/time, 5: consensus,
    # 6-7: compression ratio vs epochs/time, 8: number of links
    axes = grid.ravel()
    heatmaps = []
    print(f"{'run':55s} {'best':>8s} {'loss':>7s} {'epochs':>7s} {'time(s)':>10s} {'traffic(MB)':>12s}")
    labels = short_labels(a.files)
    for path, label in zip(a.files, labels):
        run, hist = load(path)
        ep = [h["epoch"] for h in hist]
        acc = [h.get(a.metric, float("nan")) for h in hist]
        axes[0].plot(ep, acc, label=label, marker="o", markersize=3)
        axes[1].plot([h["time"] for h in hist], acc, label=label, marker="o", markersize=3)
        axes[2].plot([h["traffic_MB"] for h in hist], acc, label=label, marker="o", markersize=3)
        full = run["history"]
        loss = smooth([h["loss"] for h in full], a.smooth)
        axes[3].plot([h["epoch"] for h in full], loss, label=label)
        axes[4].plot([h["time"] for h in full], loss, label=label)
        axes[5].plot([h["epoch"] for h in full], [h["consensus_true"] for h in full], label=label)
        r_mean = smooth([h["ratio_mean"] for h in full], a.smooth)
        r_min = smooth([h["ratio_min"] for h in full], a.smooth)
        r_max = smooth([h["ratio_max"] for h in full], a.smooth)
        for ax, x in ((axes[6], [h["epoch"] for h in full]), (axes[7], [h["time"] for h in full])):
            line, = ax.plot(x, r_mean, label=label)
            ax.fill_between(x, r_min, r_max, color=line.get_color(), alpha=0.15, linewidth=0)
        axes[8].plot([h["epoch"] for h in full], smooth([h["edges"] for h in full], a.smooth), label=label)
        if all("ratios" in h for h in full):
            heatmaps.append((label, np.array([h["ratios"] for h in full]).T, full[-1]["epoch"]))
        reach = next((h for h in hist if a.target and h.get(a.metric, 0) >= a.target), None)
        cols = (f"{reach['epoch']:7.1f} {reach['time']:10.0f} {reach['traffic_MB']:12.0f}"
                if reach else f"{'-':>7s} {'-':>10s} {'-':>12s}")
        print(f"{label:55s} {max(acc):8.4f} {loss[-1]:7.4f} {cols}")
    xlabels = ["epochs", "simulated time (s)", "traffic (MB)", "epochs", "simulated time (s)", "epochs",
               "epochs", "simulated time (s)", "epochs"]
    for ax, xl in zip(axes, xlabels):
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    for ax in axes[:3]:
        ax.set_ylabel(a.metric)
    loss_label = "training loss" + (f" (moving avg, {a.smooth} rounds)" if a.smooth > 1 else "")
    for ax in axes[3:5]:
        ax.set_ylabel(loss_label)
        if a.log_loss:
            ax.set_yscale("log")
    axes[5].set_ylabel("true consensus distance")
    avg = f" (moving avg, {a.smooth} rounds)" if a.smooth > 1 else ""
    for ax in axes[6:8]:
        ax.set_ylabel("compression ratio (mean, min-max band)")
        if avg:
            ax.set_title(f"moving avg, {a.smooth} rounds", fontsize=8)
        ax.set_ylim(0, 1.05)
    axes[8].set_ylabel("number of links" + avg)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.output, dpi=120)
    print(f"saved {a.output}")

    if heatmaps:
        hfig, haxes = plt.subplots(len(heatmaps), 1, figsize=(14, 2.2 + 0.18 * heatmaps[0][1].shape[0] * len(heatmaps)),
                                   squeeze=False)
        for ax, (label, R, last_epoch) in zip(haxes[:, 0], heatmaps):
            im = ax.imshow(R, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                           interpolation="nearest", extent=(0, last_epoch, -0.5, R.shape[0] - 0.5))
            ax.set_title(label, fontsize=9)
            ax.set_ylabel("worker")
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            hfig.colorbar(im, ax=ax, label="compression ratio", pad=0.01)
        haxes[-1, 0].set_xlabel("epochs")
        hfig.tight_layout()
        stem, ext = os.path.splitext(a.output)
        hpath = f"{stem}_ratios{ext or '.png'}"
        hfig.savefig(hpath, dpi=120)
        print(f"saved {hpath}")


if __name__ == "__main__":
    main()
