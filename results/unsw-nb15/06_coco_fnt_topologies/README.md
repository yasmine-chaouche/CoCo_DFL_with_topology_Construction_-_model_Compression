# UNSW-NB15 — CoCo with a fixed topology (FNT): ring / fully connected / star

**Algorithm:** CoCo's LP chooses one ratio per worker every round, but the topology is fixed
(`fnt-ring`, `fnt-fc`, `fnt-star`; `fnt-star` was added for this experiment). No `--ratio`.

12 runs = 3 topologies × {{IID, non-IID α = 0.1}} × H ∈ {{1, 2}}. In the merged figures,
colour = topology (ring blue, fully connected orange, star green), solid = H = 2, dashed = H = 1;
left panel IID, right panel non-IID.

```bash
OUT=results/unsw-nb15/06_coco_fnt_topologies
run() { python main.py --dataset unsw-nb15 --out-dir $OUT --log-every 1000 "$@"; }
for algo in fnt-ring fnt-fc fnt-star; do
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 2 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=1 run --algo $algo --hops 2 --partition iid &
  CUDA_VISIBLE_DEVICES=2 run --algo $algo --hops 1 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 1 --partition iid &
  wait
done
python plot_topologies.py results/unsw-nb15/06_coco_fnt_topologies/*.json --merge-hops --metric accuracy
python plot_topologies.py results/unsw-nb15/06_coco_fnt_topologies/*.json --merge-hops --metric accuracy --x time
python plot_topologies.py results/unsw-nb15/06_coco_fnt_topologies/*.json --merge-hops --metric consensus_true
python plot_topologies.py results/unsw-nb15/06_coco_fnt_topologies/*.json --merge-hops --metric consensus_true --x time
python plot_topologies.py results/unsw-nb15/06_coco_fnt_topologies/*.json --merge-hops --metric loss
python plot.py results/unsw-nb15/06_coco_fnt_topologies/*_fnt-ring[-_]*.json -o fig_grid_ring.png     # same for fnt-fc / fnt-star
```

| Figure | Content |
|---|---|
| `fig_accuracy.png` | test accuracy vs epochs |
| `fig_accuracy_vs_time.png` | test accuracy vs simulated time |
| `fig_consensus_true.png` | consensus distance after aggregation vs epochs |
| `fig_consensus_true_vs_time.png` | consensus distance after aggregation vs simulated time |
| `fig_loss.png` | training loss vs epochs |
| `fig_grid_<topology>.png` (+ `_ratios`) | `plot.py` 3×3 grid and ratio per worker × round |

⚠️ **Ring and star are not really "adapted":** D_max (Eq. 28) is unreachable with these sparse
topologies (D_max / (Σ D_ij / m²) ≈ 0.13-0.16 for the ring, 0.17-0.39 for the star, while ≈ 0.9 is
the minimum reachable), the LP is infeasible and the code falls back to r = 1 for every worker in
99 % of rounds (`coco/planner.py`, "best effort" branch — not described in the paper). Only
`fnt-fc` has adapted ratios (mean 0.20-0.41). A larger `--dmax-scale` (≈ 6-7 ring, 3-5 star) would be
needed for real adaptation.
