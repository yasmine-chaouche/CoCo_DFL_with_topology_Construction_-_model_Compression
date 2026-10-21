# UNSW-NB15 — Choco, fixed topology, r = 0.1

**Algorithm:** Choco where topology and compression ratio are both fixed (nothing is optimised),
r = 0.1 identical for all workers, RandomK mask different per worker, round and hop.
Topologies: `choco` = ring, `choco-fc` = fully connected, `choco-star` = star (worker 0 = centre).

12 runs = 3 topologies × {{IID, non-IID α = 0.1}} × H ∈ {{1, 2}}. In the merged figures,
colour = topology (ring blue, fully connected orange, star green), solid = H = 2, dashed = H = 1;
left panel IID, right panel non-IID.

```bash
OUT=results/unsw-nb15/04_choco_topologies_r0.1
run() { python main.py --dataset unsw-nb15 --ratio 0.1 --out-dir $OUT --log-every 1000 "$@"; }
for algo in choco choco-fc choco-star; do
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 2 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=1 run --algo $algo --hops 2 --partition iid &
  CUDA_VISIBLE_DEVICES=2 run --algo $algo --hops 1 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 1 --partition iid &
  wait
done
python plot_topologies.py results/unsw-nb15/04_choco_topologies_r0.1/*.json --merge-hops --metric accuracy
python plot_topologies.py results/unsw-nb15/04_choco_topologies_r0.1/*.json --merge-hops --metric accuracy --x time
python plot_topologies.py results/unsw-nb15/04_choco_topologies_r0.1/*.json --merge-hops --metric consensus_true
python plot_topologies.py results/unsw-nb15/04_choco_topologies_r0.1/*.json --merge-hops --metric consensus_true --x time
python plot_topologies.py results/unsw-nb15/04_choco_topologies_r0.1/*.json --merge-hops --metric loss
python plot.py results/unsw-nb15/04_choco_topologies_r0.1/*_choco0.1[-_]*.json -o fig_grid_ring.png     # same for choco-fc / choco-star
```

| Figure | Content |
|---|---|
| `fig_accuracy.png` | test accuracy vs epochs |
| `fig_accuracy_vs_time.png` | test accuracy vs simulated time |
| `fig_consensus_true.png` | consensus distance after aggregation vs epochs |
| `fig_consensus_true_vs_time.png` | consensus distance after aggregation vs simulated time |
| `fig_loss.png` | training loss vs epochs |
| `fig_grid_ring.png`, `fig_grid_fc.png`, `fig_grid_star.png` | `plot.py` 3×3 grid per topology |

**Observations (this conversation):** consensus distance fully connected < ring < star; H = 2 lowers
it for every topology. IID accuracy ≈ 0.72 for all; non-IID: fully connected ≈ 0.50 (H = 2) / 0.45
(H = 1), ring ≈ 0.39 / 0.36, star ≈ 0.35. Fully connected costs ≈ 10× the traffic and ≈ 5× the time of the ring.
