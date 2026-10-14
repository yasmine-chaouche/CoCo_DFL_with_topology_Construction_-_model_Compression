#!/usr/bin/env bash
# CoCo at different compression levels. CoCo chooses its ratios itself, so the
# compression level is controlled indirectly:
#   rmax : cap the ratios CoCo may choose, r in [r_min, r_max]
#   dmax : scale the consensus threshold D_max (larger = looser = more compression)
#
# Usage: bash sweep_coco.sh DATASET GPU MODE "VALUES" [extra main.py options...]
#   bash sweep_coco.sh unsw-nb15 0 rmax                      # r_max in 0.02 0.05 0.1 0.2 0.5 1
#   bash sweep_coco.sh unsw-nb15 1 dmax "0.25 0.5 1 2 4 8"
#   bash sweep_coco.sh unsw-nb15 2 rmax "0.05 0.2 1" --partition dirichlet --alpha 0.5
# Then: python plot_ratio.py results/coco_rmax/*.json
set -e
DS=${1:-unsw-nb15}
export CUDA_VISIBLE_DEVICES=${2:-0}
MODE=${3:-rmax}
case $MODE in
  rmax) VALUES=${4:-"0.02 0.05 0.1 0.2 0.5 1"}; FLAG=--r-max ;;
  dmax) VALUES=${4:-"0.25 0.5 1 2 4 8"};       FLAG=--dmax-scale ;;
  *) echo "MODE must be rmax or dmax"; exit 1 ;;
esac
shift $(( $# < 4 ? $# : 4 ))
OUT=${OUT_DIR:-results/coco_$MODE}

for v in $VALUES; do
  python main.py --dataset "$DS" --algo coco $FLAG "$v" --out-dir "$OUT" "$@"
done
