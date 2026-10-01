"""Plot accuracy vs epochs / simulated time / traffic for result files.

  python plot.py results/cifar10_*.json --target 0.87 -o cifar10.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path) as f:
        run = json.load(f)
    hist = [h for h in run["history"] if "accuracy" in h]
    return run, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--target", type=float, default=None, help="target accuracy for the summary table")
    p.add_argument("-o", "--output", default="results.png")
    a = p.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    print(f"{'run':55s} {'best acc':>8s} {'epochs':>7s} {'time(s)':>10s} {'traffic(MB)':>12s}")
    for path in a.files:
        run, hist = load(path)
        label = os.path.basename(path).replace(".json", "")
        ep = [h["epoch"] for h in hist]
        acc = [h["accuracy"] for h in hist]
        axes[0].plot(ep, acc, label=label)
        axes[1].plot([h["time"] for h in hist], acc, label=label)
        axes[2].plot([h["traffic_MB"] for h in hist], acc, label=label)
        full = run["history"]
        axes[3].plot([h["epoch"] for h in full], [h["consensus_true"] for h in full], label=label)
        reach = next((h for h in hist if a.target and h["accuracy"] >= a.target), None)
        cols = (f"{reach['epoch']:7.1f} {reach['time']:10.0f} {reach['traffic_MB']:12.0f}"
                if reach else f"{'-':>7s} {'-':>10s} {'-':>12s}")
        print(f"{label:55s} {max(acc):8.4f} {cols}")
    for ax, xl in zip(axes, ["epochs", "simulated time (s)", "traffic (MB)", "epochs"]):
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[3].set_ylabel("true consensus distance")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.output, dpi=120)
    print(f"saved {a.output}")


if __name__ == "__main__":
    main()
