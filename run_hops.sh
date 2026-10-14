#!/usr/bin/env bash
# One hop vs two hops (gossip steps) per round, on IID and non-IID (Dirichlet alpha) data.
#   * FCR r=0.1, 1 hop vs 2 hops          -> effect of the second hop at equal compression
#   * FCR r=0.2, 1 hop                    -> same traffic per round as 2 hops at r=0.1
#   * CoCo, 1 hop vs 2 hops               -> full adaptive algorithm
# Usage: bash run_hops.sh [dataset] [alpha]      (uses GPUs 0, 1, 2)
# Then:  python plot.py results/hops/*dir0.1*.json -o figures/hops/hops_dir0.1.png
DS=${1:-unsw-nb15}
ALPHA=${2:-0.1}
OUT=${OUT_DIR:-results/hops}
run() { CUDA_VISIBLE_DEVICES=$1 python main.py --dataset "$DS" --out-dir "$OUT" "${@:2}" --log-every 1000; }
for split in "--partition iid" "--partition dirichlet --alpha $ALPHA"; do
  run 0 --algo fcr --ratio 0.1 --hops 1 $split &
  run 1 --algo fcr --ratio 0.1 --hops 2 $split &
  run 2 --algo fcr --ratio 0.2 --hops 1 $split &
  wait
  run 0 --algo coco --hops 1 $split &
  run 1 --algo coco --hops 2 $split &
  wait
done
