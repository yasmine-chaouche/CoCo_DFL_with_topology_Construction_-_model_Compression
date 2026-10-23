# Runs the 12 runs of a topology scenario (3 topologies x {IID, non-IID} x H in {1, 2}) for several seeds.
# The 12 runs of one seed start together (4 per GPU), then the next seed.
# usage: bash run_seeds.sh OUT_DIR "SEEDS" [main.py options...]
#   e.g. bash run_seeds.sh results/unsw-nb15/07_choco_r0.5_mask_study/d_mask_fixed_global "1 2 3 4" --ratio 0.5 --fixed-mask
OUT=$1; SEEDS=$2; shift 2; EXTRA=("$@")
run() { python main.py --dataset unsw-nb15 --out-dir "$OUT" --log-every 1000 "${EXTRA[@]}" "$@"; }
for seed in $SEEDS; do
  i=0
  for algo in choco choco-fc choco-star; do
    for hops in 1 2; do
      for part in "--partition iid" "--partition dirichlet --alpha 0.1"; do
        CUDA_VISIBLE_DEVICES=$((i % 3)) run --algo $algo --seed $seed --hops $hops $part &
        i=$((i + 1))
      done
    done
  done
  wait
  echo "=== seed $seed done ($(date +%H:%M))"
done