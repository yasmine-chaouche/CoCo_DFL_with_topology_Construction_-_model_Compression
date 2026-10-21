#!/usr/bin/env bash
# Reproduces the main comparisons of Section 5 (Tables 2-3, Figs. 3-5, 9-12).
# Usage: bash run_experiments.sh [cifar10|cifar100|unsw-nb15] [gpu_id]
set -e
DS=${1:-cifar10}
export CUDA_VISIBLE_DEVICES=${2:-0}
OUT=${OUT_DIR:-results/$DS/paper_reproduction}
# NIID entries are "--non-iid p" levels (paper schemes) or Dirichlet alphas (unsw-nb15)
if [ "$DS" = cifar10 ]; then BASE="dcd:0.1 choco:0.5 saps:0.2"; NIID="10 20 40 60 80"; FLAG="--non-iid"
elif [ "$DS" = cifar100 ]; then BASE="dcd:0.5 choco:1.0 saps:0.2"; NIID="0 10 20 30 40"; FLAG="--non-iid"
else BASE="dcd:0.1 choco:0.5 saps:0.2"; NIID="1.0 0.5 0.1"; FLAG="--partition dirichlet --alpha"; fi

# overall performance (IID)
python main.py --dataset $DS --out-dir $OUT --algo coco
python main.py --dataset $DS --out-dir $OUT --algo lct
for b in $BASE; do python main.py --dataset $DS --out-dir $OUT --algo ${b%:*} --ratio ${b#*:}; done

# non-IID levels
for p in $NIID; do
  python main.py --dataset $DS --out-dir $OUT --algo coco $FLAG $p
  for b in $BASE; do python main.py --dataset $DS --out-dir $OUT --algo ${b%:*} --ratio ${b#*:} $FLAG $p; done
done

# ablations: fixed compression ratio / fixed topology
python main.py --dataset $DS --out-dir $OUT --algo fcr --ratio 0.1
python main.py --dataset $DS --out-dir $OUT --algo fcr --ratio 1.0
python main.py --dataset $DS --out-dir $OUT --algo fnt-ring
python main.py --dataset $DS --out-dir $OUT --algo fnt-fc

# network environments
for net in heter_static homo_static; do python main.py --dataset $DS --out-dir $OUT --algo coco --network $net; done
python main.py --dataset $DS --out-dir $OUT --algo coco --network heter_static --track-workers --name ${DS}_traffic_static
python main.py --dataset $DS --out-dir $OUT --algo coco --network heter_static --track-workers --swap-at 100 --name ${DS}_traffic_swap

# TopK compressor
python main.py --dataset $DS --out-dir $OUT --algo coco --compressor topk
