# Results

One folder per experiment: `results/<dataset>/<NN>_<experiment>/`. Each folder holds:

| File | What it is |
|---|---|
| `README.md` | the question, the settings, the commands that produced the runs and the figures |
| `*.json` | final results: one file per run (`config` + per-round `history`), written by `main.py` |
| `fig_*.png` | figures made from the `.json` files of the same folder |
| `logs/` | intermediate files (console logs), when they were kept |

New runs go to `results/<dataset>/unsorted/` by default (`main.py --out-dir` changes it).
Once an experiment is finished, move it to the next numbered folder and add a `README.md`.

`MOVES.tsv` lists where every file was before the 2026-10-20 reorganisation (old path → new path).
The `config.out_dir` field inside older `.json` files still shows the old folder.

## File names

`<dataset>-<task>_<model>_<algo><ratio>[-<H>hop][-<option>]_<split>_<network>_s<seed>.json`

| Part | Values |
|---|---|
| `<algo>` | `coco`, `fcr`, `fnt-ring/fc/star` (CoCo variants); `choco` (ring), `choco-fc`, `choco-star`, `dcd`, `saps`, `lct` |
| `<ratio>` | fixed compression ratio (absent for CoCo / FNT / LCT); `-r0.1-0.9` = one fixed ratio per worker in [0.1, 0.9] |
| `-<H>hop` | H gossip steps per round (absent = 1) |
| `-<option>` | `-roundmask`, `-nodemask`, `-fixedmask` (RandomK mask rule), `-topk` (TopK compressor) |
| `<split>` | `iid`, `dir<α>` (Dirichlet non-IID), `niid<p>` (paper schemes) |
| `<network>` | `heter_dynamic` (default), `heter_static`, `homo_static` |

Figures: `fig_<metric>[_vs_time|_vs_traffic].png` (x-axis = epochs unless stated),
`fig_grid_<topology>.png` (3×3 grid of `plot.py`), `*_ratios.png` (ratio per worker × round).

## Index

Common settings for UNSW-NB15: multiclass task, MLP (d = 83 594 parameters), 20 workers,
200 rounds (≈ 36.5 epochs), τ = 50 local steps, batch 32, lr 0.05, network `heter_dynamic`,
seed 0, RandomK. "non-IID" = Dirichlet α (0.1 unless stated).

| Folder | Question |
|---|---|
| [cifar10/01_coco_smoke_test](cifar10/01_coco_smoke_test) | first CoCo run on CIFAR-10 (VGG9) |
| [unsw-nb15/01_paper_reproduction](unsw-nb15/01_paper_reproduction) | Section 5 of the paper on UNSW-NB15: CoCo vs DCD / Choco / SAPS / LCT, ablations, network environments |
| [unsw-nb15/02_ratio_sweep_fcr](unsw-nb15/02_ratio_sweep_fcr) | accuracy vs compression ratio (FCR r = 0.02 … 1), 4 data splits, CoCo as reference |
| [unsw-nb15/03_hops_fcr_vs_coco](unsw-nb15/03_hops_fcr_vs_coco) | 1 vs 2 gossip steps per round, FCR and CoCo |
| [unsw-nb15/04_choco_topologies_r0.1](unsw-nb15/04_choco_topologies_r0.1) | Choco, fixed topology (ring / fully connected / star), r = 0.1, H = 1 / 2 |
| [unsw-nb15/05_choco_topologies_r1.0](unsw-nb15/05_choco_topologies_r1.0) | same as 04 without compression (r = 1) |
| [unsw-nb15/06_coco_fnt_topologies](unsw-nb15/06_coco_fnt_topologies) | CoCo with a fixed topology and adapted ratios (FNT ring / fc / star) |
| [unsw-nb15/07_choco_r0.5_mask_study](unsw-nb15/07_choco_r0.5_mask_study) | same as 04 at r = 0.5, four rules for the RandomK mask |
| [unsw-nb15/08_choco_ratio_per_worker](unsw-nb15/08_choco_ratio_per_worker) | same as 04 with a different fixed ratio per worker (r ∈ [0.1, 0.9], mean 0.5) |
