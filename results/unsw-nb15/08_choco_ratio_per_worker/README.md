# UNSW-NB15 — Choco with a different fixed ratio per worker

**Algorithm:** Choco, fixed topology (ring / fully connected / star). Each worker has its own
compression ratio, fixed for the whole run: 20 values evenly spaced in [0.1, 0.9] (mean exactly
0.5), randomly assigned to workers (`--ratio-range 0.1 0.9`, own generator → same assignment for
every topology). RandomK mask different per worker, round and hop.

12 runs = 3 topologies × {{IID, non-IID α = 0.1}} × H ∈ {{1, 2}}. In the merged figures,
colour = topology (ring blue, fully connected orange, star green), solid = H = 2, dashed = H = 1;
left panel IID, right panel non-IID.

```bash
OUT=results/unsw-nb15/08_choco_ratio_per_worker
run() { python main.py --dataset unsw-nb15 --ratio-range 0.1 0.9 --out-dir $OUT --log-every 1000 "$@"; }
for algo in choco choco-fc choco-star; do
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 2 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=1 run --algo $algo --hops 2 --partition iid &
  CUDA_VISIBLE_DEVICES=2 run --algo $algo --hops 1 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 1 --partition iid &
  wait
done
python plot_topologies.py results/unsw-nb15/08_choco_ratio_per_worker/*.json --merge-hops --metric accuracy
python plot_topologies.py results/unsw-nb15/08_choco_ratio_per_worker/*.json --merge-hops --metric accuracy --x time
python plot_topologies.py results/unsw-nb15/08_choco_ratio_per_worker/*.json --merge-hops --metric consensus_true
python plot_topologies.py results/unsw-nb15/08_choco_ratio_per_worker/*.json --merge-hops --metric consensus_true --x time
python plot_topologies.py results/unsw-nb15/08_choco_ratio_per_worker/*.json --merge-hops --metric loss
```

| Figure | Content |
|---|---|
| `fig_accuracy.png` | test accuracy vs epochs |
| `fig_accuracy_vs_time.png` | test accuracy vs simulated time |
| `fig_consensus_true.png` | consensus distance after aggregation vs epochs |
| `fig_consensus_true_vs_time.png` | consensus distance after aggregation vs simulated time |
| `fig_loss.png` | training loss vs epochs |

Note: in the star the centre (worker 0) got r = 0.65 and serves 19 links, which makes the star
runs very slow in simulated time (≈ 12 800 s for H = 2, IID).
