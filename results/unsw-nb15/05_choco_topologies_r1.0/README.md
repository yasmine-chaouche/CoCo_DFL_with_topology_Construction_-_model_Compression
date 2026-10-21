# UNSW-NB15 — Choco, fixed topology, no compression (r = 1)

**Algorithm:** same as `04_choco_topologies_r0.1` with r = 1 (every parameter is sent).
Question: does the benefit of H = 2 change when there is no compression? (Expected: with a fully
connected graph and r = 1, one step already gives the exact average, so H = 2 should not help;
ring and star should benefit.)

12 runs = 3 topologies × {{IID, non-IID α = 0.1}} × H ∈ {{1, 2}}. In the merged figures,
colour = topology (ring blue, fully connected orange, star green), solid = H = 2, dashed = H = 1;
left panel IID, right panel non-IID.

```bash
OUT=results/unsw-nb15/05_choco_topologies_r1.0
run() { python main.py --dataset unsw-nb15 --ratio 1.0 --out-dir $OUT --log-every 1000 "$@"; }
for algo in choco choco-fc choco-star; do
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 2 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=1 run --algo $algo --hops 2 --partition iid &
  CUDA_VISIBLE_DEVICES=2 run --algo $algo --hops 1 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 1 --partition iid &
  wait
done
python plot_topologies.py results/unsw-nb15/05_choco_topologies_r1.0/*.json --merge-hops --metric accuracy
python plot_topologies.py results/unsw-nb15/05_choco_topologies_r1.0/*.json --merge-hops --metric accuracy --x time
python plot_topologies.py results/unsw-nb15/05_choco_topologies_r1.0/*.json --merge-hops --metric consensus_true
python plot_topologies.py results/unsw-nb15/05_choco_topologies_r1.0/*.json --merge-hops --metric consensus_true --x time
python plot_topologies.py results/unsw-nb15/05_choco_topologies_r1.0/*.json --merge-hops --metric loss
```

| Figure | Content |
|---|---|
| `fig_accuracy.png` | test accuracy vs epochs |
| `fig_accuracy_vs_time.png` | test accuracy vs simulated time |
| `fig_consensus_true.png` | consensus distance after aggregation vs epochs |
| `fig_consensus_true_vs_time.png` | consensus distance after aggregation vs simulated time |
| `fig_loss.png` | training loss vs epochs |
