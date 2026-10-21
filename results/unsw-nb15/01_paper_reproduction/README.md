# UNSW-NB15 — reproduction of the paper's Section 5

Main comparisons of the paper (Tables 2-3, Figs. 3-5, 9-12) on UNSW-NB15, produced by
`run_experiments.sh unsw-nb15`.

| Runs | Content |
|---|---|
| `coco_*`, `choco0.5_*`, `dcd0.1_*`, `saps0.2_*` | CoCo vs baselines on IID and Dirichlet α ∈ {1, 0.5, 0.1} |
| `lct_iid` | logically centralised training (complete graph, no compression) |
| `fcr0.1_iid`, `fcr1.0_iid`, `fnt-ring_iid`, `fnt-fc_iid` | ablations: fixed ratio (FCR) / fixed topology (FNT) |
| `coco_iid_heter_static`, `coco_iid_homo_static` | network environments |
| `unsw-nb15_traffic_static`, `unsw-nb15_traffic_swap` | per-worker traffic (`--track-workers`), bandwidth swap at epoch 100 |

⚠️ `unsw-nb15-multiclass_mlp_coco_iid_heter_dynamic_s0.json` is the **TopK** run
(`config.compressor = "topk"`): the last step of `run_experiments.sh` overwrote the RandomK
CoCo IID run, because file names did not include the compressor at that time (fixed since:
TopK runs now get `-topk`). A RandomK CoCo IID run with the same settings is in
`../02_ratio_sweep_fcr/`.

```bash
OUT_DIR=results/unsw-nb15/01_paper_reproduction bash run_experiments.sh unsw-nb15 0
python plot.py results/unsw-nb15/01_paper_reproduction/*.json -o fig_unsw-nb15.png
```

| Figure | Content |
|---|---|
| `fig_unsw-nb15.png` (+ `_ratios`) | `plot.py` grid over the runs of this folder |
| `fig_coco_iid.png`, `fig_coco_dir.png`, `fig_unsw_coco_iid.png` (+ `_ratios`) | CoCo runs only |
| `fig_coco_vs_fcr.png` | CoCo vs fixed-ratio ablation |

The exact list of runs behind each figure was not recorded; the file names indicate the subset.
