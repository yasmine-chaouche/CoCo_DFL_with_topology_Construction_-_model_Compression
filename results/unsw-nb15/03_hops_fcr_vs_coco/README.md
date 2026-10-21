# UNSW-NB15 — one vs two gossip steps per round

Effect of H (gossip steps per round) on IID and Dirichlet α = 0.1, produced by `run_hops.sh`:

| Runs | Question |
|---|---|
| `fcr0.1` vs `fcr0.1-2hop` | second hop at equal compression |
| `fcr0.2` | 1 hop with the same traffic per round as 2 hops at r = 0.1 |
| `coco` vs `coco-2hop` | effect of H on the full adaptive algorithm |

```bash
OUT_DIR=results/unsw-nb15/03_hops_fcr_vs_coco bash run_hops.sh unsw-nb15 0.1
D=results/unsw-nb15/03_hops_fcr_vs_coco
python plot.py $D/*fcr*iid*.json   -o fig_iid.png
python plot.py $D/*fcr*dir0.1*.json -o fig_dir0.1.png
python plot.py $D/*coco*iid*.json  -o fig_coco_iid.png
python plot.py $D/*coco*dir0.1*.json -o fig_coco_dir0.1.png
python plot.py $D/*coco*.json      -o fig_coco_all.png
```

The exact file subset behind each figure was not recorded; the commands above follow the file names.
`fig_*_ratios.png` = compression ratio per worker × round; `fig_coco_consensus_time.png` =
consensus distance of the CoCo runs vs simulated time. `logs/hops.log` = console log.
