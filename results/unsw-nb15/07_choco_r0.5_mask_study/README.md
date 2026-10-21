# UNSW-NB15 — Choco r = 0.5: effect of the RandomK mask rule

**Algorithm:** Choco, fixed topology (ring / fully connected / star), fixed ratio r = 0.5 for all
workers. Only the rule that draws the RandomK mask changes between sub-folders. In every case the
mask is identical across topologies (the seed does not depend on the topology).

| Sub-folder | Mask differs between workers | changes every round | changes between hops | Option |
|---|---|---|---|---|
| `a_mask_per_worker_round_hop` | yes | yes | yes | none (default) |
| `b_mask_per_worker_round` | yes | yes | no | `--round-mask` |
| `c_mask_fixed_per_worker` | yes | no | no | `--fixed-node-mask` |
| `d_mask_fixed_global` | no | no | no | `--fixed-mask` |

With H = 1, `a` and `b` use exactly the same seeds (only the H = 2 runs differ).

12 runs = 3 topologies × {{IID, non-IID α = 0.1}} × H ∈ {{1, 2}}. In the merged figures,
colour = topology (ring blue, fully connected orange, star green), solid = H = 2, dashed = H = 1;
left panel IID, right panel non-IID.

```bash
# one sub-folder at a time, e.g. b:
OUT=results/unsw-nb15/07_choco_r0.5_mask_study/b_mask_per_worker_round
run() { python main.py --dataset unsw-nb15 --ratio 0.5 --round-mask --out-dir $OUT --log-every 1000 "$@"; }
for algo in choco choco-fc choco-star; do
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 2 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=1 run --algo $algo --hops 2 --partition iid &
  CUDA_VISIBLE_DEVICES=2 run --algo $algo --hops 1 --partition dirichlet --alpha 0.1 &
  CUDA_VISIBLE_DEVICES=0 run --algo $algo --hops 1 --partition iid &
  wait
done
python plot_topologies.py results/unsw-nb15/07_choco_r0.5_mask_study/b_mask_per_worker_round/*.json --merge-hops --metric accuracy
python plot_topologies.py results/unsw-nb15/07_choco_r0.5_mask_study/b_mask_per_worker_round/*.json --merge-hops --metric accuracy --x time
python plot_topologies.py results/unsw-nb15/07_choco_r0.5_mask_study/b_mask_per_worker_round/*.json --merge-hops --metric consensus_true
python plot_topologies.py results/unsw-nb15/07_choco_r0.5_mask_study/b_mask_per_worker_round/*.json --merge-hops --metric consensus_true --x time
python plot_topologies.py results/unsw-nb15/07_choco_r0.5_mask_study/b_mask_per_worker_round/*.json --merge-hops --metric loss
```

Each sub-folder has the 5 figures `fig_accuracy`, `fig_accuracy_vs_time`, `fig_consensus_true`,
`fig_consensus_true_vs_time`, `fig_loss`.

**Observations (this conversation):** with a global fixed mask (`d`) the 50 % of parameters never
sent drift apart, so the consensus distance *increases* during training and no longer depends much on
the topology; non-IID accuracy drops to ≈ 0.34-0.35 for every topology (star slightly best).
