# UNSW-NB15 — accuracy vs compression ratio (FCR sweep)

FCR (CoCo topology search, fixed identical ratio) with r ∈ {0.02, 0.05, 0.1, 0.2, 0.5, 1.0},
plus CoCo (adaptive ratios) as reference, on 4 data splits: IID and Dirichlet α ∈ {1, 0.5, 0.1}.
28 runs, H = 1. Produced by `sweep_ratio.sh` (one call per split).

```bash
for s in "--partition iid" "--partition dirichlet --alpha 1" "--partition dirichlet --alpha 0.5" \
         "--partition dirichlet --alpha 0.1"; do
  OUT_DIR=results/unsw-nb15/02_ratio_sweep_fcr bash sweep_ratio.sh unsw-nb15 0 fcr "0.02 0.05 0.1 0.2 0.5 1.0" $s
done
D=results/unsw-nb15/02_ratio_sweep_fcr
python plot_ratio.py $D/*_iid_*.json -o fig_ratio_sweep_iid.png
python plot_ratio.py $D/*.json        -o fig_ratio_sweep_all_splits.png
```

| Figure | Content |
|---|---|
| `fig_ratio_sweep_iid.png` | accuracy, F1, consensus, time, traffic vs ratio (IID only) |
| `fig_ratio_sweep_all_splits.png` | same, one curve per data split |
| `*_consensus.png` | consensus distance over training, coloured by ratio |

`logs/` holds the console logs of the three Dirichlet sweeps.
