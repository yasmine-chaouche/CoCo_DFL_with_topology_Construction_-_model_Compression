#!/usr/bin/env bash
# Impact of the compression ratio on accuracy: one run per (algorithm, ratio), plus CoCo
# (adaptive ratios) as a reference.
#
# Usage: bash sweep_ratio.sh DATASET GPU "ALGOS" "RATIOS" [extra main.py options...]
#   bash sweep_ratio.sh unsw-nb15 0
#   bash sweep_ratio.sh unsw-nb15 1 "choco dcd" "0.01 0.1 0.5 1.0" --partition dirichlet --alpha 0.5
#   bash sweep_ratio.sh cifar10 2 "choco" "0.1 0.2 0.5 1.0" --epochs 50
# Then: python plot_ratio.py results/ratio_sweep/*.json
set -e
DS=${1:-unsw-nb15}
export CUDA_VISIBLE_DEVICES=${2:-0}
ALGOS=${3:-"choco dcd fcr"}
RATIOS=${4:-"0.01 0.05 0.1 0.2 0.5 1.0"}
shift $(( $# < 4 ? $# : 4 ))
OUT=${OUT_DIR:-results/ratio_sweep}

for algo in $ALGOS; do
  for r in $RATIOS; do
    python main.py --dataset "$DS" --algo "$algo" --ratio "$r" --out-dir "$OUT" "$@"
  done
done
python main.py --dataset "$DS" --algo coco --out-dir "$OUT" "$@"
