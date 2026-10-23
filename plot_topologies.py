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
Several seeds of the same run (same H, split and topology) are drawn as mean (line)
± std (shaded band; with --merge-hops, hatched for H=1); with --x time or traffic the seeds are
first interpolated on a common grid.
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
FAMILIES = {"choco": "Choco topology ({r})","fnt": "CoCo, fixed topology (r adapted in [{r}])"}
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


def aggregate(curves):
    """Mean and std (None for a single run) of the seeds of one run, on a common x."""
    if len(curves) == 1:
        x, y = curves[0]
        return x, y, None
    n = min(len(x) for x, _ in curves)
    x0 = curves[0][0][:n]
    if all(np.allclose(x[:n], x0) for x, _ in curves):  # same x for every seed (epochs)
        Y = np.stack([y[:n] for _, y in curves])
    else:  # simulated time / traffic differ between seeds: interpolate on the shared range
        x0 = np.linspace(max(x[0] for x, _ in curves), min(x[-1] for x, _ in curves), n)
        Y = np.stack([np.interp(x0, x, y) for x, y in curves])
    return x0, Y.mean(axis=0), Y.std(axis=0, ddof=1)


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

    runs = {}  # (hops, split) -> {topology: [(x, values), ...]}, one entry per seed
    seeds, finals = {}, {}  # (hops, split, topology) -> seeds / (time, traffic) of every seed
    families, ratios = set(), set()
    print(f"{'H':>2s} {'split':32s} {'topology':16s} {'seed':>4s} {'epochs':>7s} {'time(s)':>10s} {'traffic(MB)':>12s}")
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
        if family == "fnt":
            ratios.add(f"{cfg['r_min']:g}, {cfg['r_max']:g}")
        elif cfg.get("ratio_range"):  # --ratio-range: cfg["ratio"] is the unused default
            ratios.add("r fixed per worker, in [{:g}, {:g}]".format(*cfg["ratio_range"]))
        else:
            ratios.add(f"r fixed = {cfg['ratio']:g}")
        key = (cfg.get("hops", 1), split_name(cfg), topo)
        if cfg["seed"] in seeds.get(key, []):
            raise SystemExit(f"{path}: seed {cfg['seed']} given twice for H={key[0]}, {key[1]}, {topo}")
        seeds.setdefault(key, []).append(cfg["seed"])
        runs.setdefault(key[:2], {}).setdefault(topo, []).append((ep, y))
        last = run["history"][-1]
        finals.setdefault(key, []).append((last["time"], last["traffic_MB"]))
        print(f"{key[0]:2d} {key[1]:32s} {topo:16s} {cfg['seed']:4d} "
              f"{last['epoch']:7.1f} {last['time']:10.0f} {last['traffic_MB']:12.0f}")

    if len(families) > 1:
        raise SystemExit("mix of Choco and CoCo-FNT runs: plot one family at a time")
    title = FAMILIES[families.pop()].format(r=" / ".join(sorted(ratios)))
    n_seeds = sorted({len(s) for s in seeds.values()})
    if n_seeds != [1]:
        if len(n_seeds) > 1:
            print(f"warning: not the same number of seeds for every curve ({n_seeds})")
        title += f"\nmean ± std over {'/'.join(map(str, n_seeds))} seeds"
        print(f"\n{'H':>2s} {'split':32s} {'topology':16s} {'seeds':>5s} {'time(s) mean ± std':>20s} "
              f"{'traffic(MB) mean ± std':>24s}")
        for (h, s, topo), f in sorted(finals.items()):
            t, tr = np.array(f).T
            sd = (lambda v: v.std(ddof=1)) if len(f) > 1 else (lambda v: 0.0)
            print(f"{h:2d} {s:32s} {topo:16s} {len(f):5d} {t.mean():12.0f} ± {sd(t):<6.0f} "
                  f"{tr.mean():14.0f} ± {sd(tr):<6.0f}")
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
                    ep, y, std = aggregate(runs[(h, s)][label])
                    if a.merge_hops:
                        ax.plot(ep, y, color=color, linestyle=hop_styles[h], linewidth=2 if h == hops[0] else 1.6,
                                label=f"{label}, H={h}")
                    else:
                        ax.plot(ep, y, color=color, linestyle=ls, linewidth=2, label=label)
                    if std is not None:  # on a log axis, keep the lower edge of the band above 0
                        lo = np.maximum(y - std, y * 0.01) if log else y - std
                        if a.merge_hops and h != hops[0]:  # hatched band for the dashed H: overlaps stay readable
                            ax.fill_between(ep, lo, y + std, facecolor="none", edgecolor=color, alpha=0.45,
                                            hatch="////", linewidth=0)
                        else:
                            ax.fill_between(ep, lo, y + std, color=color, alpha=0.15, linewidth=0)
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
        if n_seeds != [1]:  # std band of each H under its line: filled for the solid H, hatched for the others
            from matplotlib.patches import Patch
            bands = [Patch(facecolor="#52514e", alpha=0.15, linewidth=0)] + [
                Patch(facecolor="none", edgecolor="#52514e", alpha=0.45, hatch="////", linewidth=0)] * (len(hops) - 1)
            hop = list(zip(bands, hop))
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
