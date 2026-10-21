"""Compare fixed topologies (ring / fully connected / star), one panel per scenario.

  python plot_topologies.py results/unsw-nb15/04_choco_topologies_r0.1/*.json --merge-hops --metric accuracy
      -> results/unsw-nb15/04_choco_topologies_r0.1/fig_accuracy.png

Works for Choco (choco / choco-fc / choco-star: fixed ratio) or for CoCo with a fixed
topology (fnt-ring / fnt-fc / fnt-star: adaptive ratios); pass one family at a time.

Panels: rows = gossip steps per round (H), columns = data split (IID / non-IID).
Each panel has one curve per topology, with the same colour and line style everywhere.
--merge-hops draws H=1 and H=2 in the same panel (colour = topology, line style = H),
giving one row: IID and non-IID side by side.
--x picks the horizontal axis (epochs, simulated time or cumulative traffic); the final
time and traffic of every run are also printed as a table.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

STYLES = {  # topology -> (colour, line style)
    "ring": ("#2a78d6", "-"),
    "fully connected": ("#eb6834", "--"),
    "star": ("#1baf7a", "-."),
}
TOPOLOGIES = {  # algo -> (topology, family)
    "choco": ("ring", "choco"), "choco-fc": ("fully connected", "choco"), "choco-star": ("star", "choco"),
    "fnt-ring": ("ring", "fnt"), "fnt-fc": ("fully connected", "fnt"), "fnt-star": ("star", "fnt"),
}
FAMILIES = {"choco": "Choco topology (r fixed = {r})", "fnt": "CoCo, fixed topology (r adapted in [{r}])"}
METRICS = {
    "consensus_true": ("consensus distance", True),
    "consensus_true_pre": ("consensus distance (before aggregation)", True),
    "accuracy": ("test accuracy", False),
    "f1_macro": ("macro-F1", False),
    "loss": ("training loss", True),
    "time": ("simulated time (s)", False),
    "traffic_MB": ("cumulative traffic (MB)", True),
}
NO_SMOOTH = ("accuracy", "f1_macro", "time", "traffic_MB")  # per-epoch or cumulative values
XAXES = {"epoch": "epochs", "time": "simulated time (s)", "traffic_MB": "traffic (MB)"}


def smooth(values, window):
    """Trailing moving average."""
    v = np.asarray(values, dtype=float)
    if window <= 1:
        return v
    c = np.cumsum(np.insert(v, 0, 0.0))
    idx = np.arange(1, len(v) + 1)
    n = np.minimum(idx, window)
    return (c[idx] - c[idx - n]) / n


def split_name(cfg):
    if cfg["partition"] == "iid":
        return "IID"
    if cfg["partition"] == "dirichlet":
        return f"non-IID (Dirichlet α={cfg['alpha']:g})"
    return f"non-IID ({cfg['non_iid']:g})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--metric", default="consensus_true", choices=list(METRICS))
    p.add_argument("--x", default="epoch", choices=list(XAXES), help="horizontal axis")
    p.add_argument("--linear-x", action="store_true", help="linear traffic axis (default: log)")
    p.add_argument("--merge-hops", action="store_true",
                   help="one panel per data split, H=1 and H=2 in the same panel")
    p.add_argument("--smooth", type=int, default=5, help="moving average (rounds)")
    p.add_argument("-o", "--output", default=None,
                   help="default: fig_<metric>[_vs_<x>].png (fig_<metric>_by_hops... without --merge-hops) "
                        "next to the first input .json; a bare file name also goes there")
    a = p.parse_args()
    if a.output is None:
        a.output = (f"fig_{a.metric}" + ("" if a.merge_hops else "_by_hops")
                    + ("" if a.x == "epoch" else f"_vs_{a.x.replace('_MB', '')}") + ".png")
    # a bare file name goes next to the result files, so figures and .json stay together
    if not os.path.dirname(a.output):
        a.output = os.path.join(os.path.dirname(a.files[0]) or ".", a.output)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    runs = {}  # (hops, split) -> {topology: (x, values)}
    families, ratios = set(), set()
    print(f"{'H':>2s} {'split':32s} {'topology':16s} {'epochs':>7s} {'time(s)':>10s} {'traffic(MB)':>12s}")
    for path in a.files:
        with open(path) as f:
            run = json.load(f)
        cfg = run["config"]
        if cfg["algo"] not in TOPOLOGIES:
            print(f"skipped {path} (algo {cfg['algo']})")
            continue
        hist = [h for h in run["history"] if a.metric in h]
        ep = np.array([h[a.x] for h in hist])
        y = np.array([h[a.metric] for h in hist], dtype=float)
        if a.metric not in NO_SMOOTH:
            y = smooth(y, a.smooth)
        topo, family = TOPOLOGIES[cfg["algo"]]
        families.add(family)
        ratios.add(f"{cfg['ratio']:g}" if family == "choco" else f"{cfg['r_min']:g}, {cfg['r_max']:g}")
        runs.setdefault((cfg.get("hops", 1), split_name(cfg)), {})[topo] = (ep, y)
        last = run["history"][-1]
        print(f"{cfg.get('hops', 1):2d} {split_name(cfg):32s} {topo:16s} "
              f"{last['epoch']:7.1f} {last['time']:10.0f} {last['traffic_MB']:12.0f}")

    if len(families) > 1:
        raise SystemExit("mix of Choco and CoCo-FNT runs: plot one family at a time")
    title = FAMILIES[families.pop()].format(r=" / ".join(sorted(ratios)))
    hops = sorted({h for h, _ in runs}, reverse=True)
    splits = sorted({s for _, s in runs}, key=lambda s: (s != "IID", s))
    ylabel, log = METRICS[a.metric]
    rows = [hops] if a.merge_hops else [[h] for h in hops]  # the H values drawn in each row
    hop_styles = dict(zip(hops, ["-", "--", ":"]))  # with --merge-hops: line style = H
    fig, axes = plt.subplots(len(rows), len(splits),
                             figsize=(7 * len(splits), 5) if a.merge_hops else (6 * len(splits), 4 * len(hops)),
                             sharex="col", sharey="col", squeeze=False)
    for r, row in enumerate(rows):
        for c, s in enumerate(splits):
            ax = axes[r, c]
            for h in row:
                for label, (color, ls) in STYLES.items():
                    if label not in runs.get((h, s), {}):
                        continue
                    ep, y = runs[(h, s)][label]
                    if a.merge_hops:
                        ax.plot(ep, y, color=color, linestyle=hop_styles[h], linewidth=2 if h == hops[0] else 1.6,
                                label=f"{label}, H={h}")
                    else:
                        ax.plot(ep, y, color=color, linestyle=ls, linewidth=2, label=label)
            ax.set_title(s if a.merge_hops else f"H={row[0]}, {s}", fontsize=11)
            if log:
                ax.set_yscale("log")
            else:
                ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
                ax.grid(alpha=0.12, which="minor", axis="y", color="#b0b0b0")
            if a.x == "traffic_MB" and not a.linear_x:
                ax.set_xscale("log")
            ax.grid(alpha=0.25, which="major", color="#b0b0b0")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == len(rows) - 1:
                ax.set_xlabel(XAXES[a.x])
            if c == 0:
                ax.set_ylabel(ylabel + (f"\n(moving avg, {a.smooth} rounds)"
                                        if a.smooth > 1 and a.metric not in NO_SMOOTH else ""))
    if log:  # plain numbers (2, 3, 5, 10...) instead of 2x10^0, denser when the range is narrow
        for ax in axes[0]:
            lo, hi = ax.get_ylim()
            decades = np.log10(hi / lo)
            subs = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9) if decades < 0.5 else (1, 1.5, 2, 3, 4, 5, 6, 8) if decades < 1.5 else (1, 2, 5) if decades < 3 else (1,)
            ticks = [m * 10.0 ** e for e in range(-4, 8) for m in subs]
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))  # shared by the whole column
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
    if a.merge_hops:  # two legends: colour = topology, line style = H
        from matplotlib.lines import Line2D
        topo = [Line2D([], [], color=c, linewidth=2.5) for c, _ in STYLES.values()]
        hop = [Line2D([], [], color="#52514e", linestyle=hop_styles[h], linewidth=2) for h in hops]
        fig.legend(topo, list(STYLES), loc="upper center", ncol=len(topo),
                   frameon=False, fontsize=11, title=title, bbox_to_anchor=(0.33, 1.04))
        fig.legend(hop, [f"H={h}" for h in hops], loc="upper center", ncol=len(hop), frameon=False,
                   fontsize=11, title="gossip steps per round", bbox_to_anchor=(0.75, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   fontsize=11, title=title, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.output, dpi=130, bbox_inches="tight")
    print(f"saved {a.output}")


if __name__ == "__main__":
    main()
