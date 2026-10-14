# CoCo-DFL

Implementation of **CoCo** (topology **Co**nstruction + model **Co**mpression) from
*L. Wang, Y. Xu, H. Xu, M. Chen, L. Huang, "Accelerating Decentralized Federated
Learning in Heterogeneous Edge Computing", IEEE TMC 22(9), 2023.*

Like the paper's simulation platform (Section 5.1), the edge system runs in a single process:
`m` workers train real PyTorch models on the GPU, and wall-clock time / traffic are
**simulated** from the heterogeneous, dynamic bandwidth model.

## Layout

| file | paper |
|---|---|
| `coco/planner.py` | **Algorithm 3 (CoCo)**, exact solver for the LP (27), baselines and ablations |
| `coco/consensus.py` | consensus distance: Eq. 23 (measured over compressed models), Eqs. 24-26 (Floyd-Warshall + EMA for unlinked pairs), Eq. 28 (`D_max`) |
| `coco/simulator.py` | Algorithm 1 (workers) + Algorithm 2 (coordinator): local SGD (Eq. 5), compression, element-wise aggregation (Eqs. 6-7), round time (Eq. 18), traffic |
| `coco/network.py` | bandwidth / computation model: Eq. 1 link bandwidth, Eqs. 15-18, connectivity (19b) |
| `coco/compression.py` | RandomK (seeded mask) and TopK sparsification |
| `coco/data.py` | CIFAR-10/100 loader (no torchvision), IID / non-IID partitions of Section 5.3, Dirichlet partition |
| `coco/unsw.py` | UNSW-NB15 intrusion-detection dataset (extension, not in the paper) |
| `coco/models.py` | VGG9 (3.49M params), ResNet9 (6.6M params), MLP for tabular data |
| `main.py` | command-line entry point |
| `plot.py` | accuracy vs epochs / time / traffic, time-to-target table |
| `run_experiments.sh` | the experiment grid of Section 5 |
| `tests/test_planner.py` | cross-checks the LP solver against `scipy.optimize.linprog` |

## Usage

```bash
pip install -r requirements.txt
python main.py --algo coco --dataset cifar10              # VGG9, 20 workers, 200 epochs
python main.py --algo coco --dataset cifar100 --non-iid 20  # ResNet9, each worker lacks 20 classes
python main.py --algo choco --ratio 0.5 --dataset cifar10   # baseline
python plot.py results/cifar10_*.json --target 0.87 -o figures/cifar10/cifar10.png
python tests/test_planner.py
```

Datasets are read from `--data-dir` (default `../data`, either `<dir>/cifar-10-batches-py`
or `<dir>/cifar10/cifar-10-batches-py`) and downloaded if missing.

## UNSW-NB15 (extension)

```bash
python main.py --dataset unsw-nb15 --algo coco                                    # IID, 10 classes
python main.py --dataset unsw-nb15 --algo coco --partition dirichlet --alpha 0.5  # non-IID
python main.py --dataset unsw-nb15 --unsw-task binary --algo choco --ratio 0.5    # normal vs attack
bash run_experiments.sh unsw-nb15 0
```

* **Data**: the official train/test partition (175,341 / 82,332 flows), read from
  `<data-dir>/unsw-nb15/` as the official CSVs (`UNSW_NB15_training-set.csv`,
  `UNSW_NB15_testing-set.csv`) or as the parquet release of Kaggle `dhoogla/unswnb15`.
  The parquet files are downloaded from a Hugging Face mirror if neither is present.
  The parquet release drops a few columns (e.g. `sttl`, `dttl`); only columns present in both files are used.
* **Features**: `proto`, `service` and `state` are one-hot encoded; numeric columns use
  `sign(x)·log1p(|x|)` and are standardised with training statistics. This gives 186 inputs.
* **Task**: `--unsw-task multiclass` (default) uses the 10 `attack_cat` classes; `binary` uses `label`.
* **Model**: MLP 186-256-128-C with BatchNorm (83.6k parameters, B = 2.7 Mb).
* **Partitions**: `iid` (default) or `--partition dirichlet --alpha a`. A smaller `a` gives more skewed
  label distributions and unequal local dataset sizes. `dominant` and `missing` also work, but
  several classes are tiny (Worms: 130 training samples).
* **Metrics**: accuracy and **macro-F1** (`f1_macro`) are logged for every dataset. Macro-F1 is the
  more meaningful metric here because the classes are very imbalanced.
* **Computation time** of τ local steps is assumed to be 0.5–1.5 s (`--comp-range`). The paper
  has no measurement for this model.
* **Defaults**: 30 epochs, lr 0.05 with 0.98 decay per epoch, τ = 50, batch 32.

`--algo`:

* `coco` – the proposed algorithm
* `dcd`, `choco` – fixed degree-5 / ring topology with a fixed, identical ratio `--ratio`
* `saps` – adaptive bandwidth-aware topology (max-bandwidth spanning tree each round) with a fixed ratio
* `lct` – logically centralized training (complete graph, no compression)
* `fcr` – CoCo with fixed compression ratio `--ratio` (topology search only)
* `fnt-ring`, `fnt-fc` – CoCo with a fixed ring / complete topology (ratio optimisation only)

Other options: `--network {heter_dynamic,heter_static,homo_static}` (Section 5.5),
`--swap-at EPOCH` (exchange bandwidth of workers 1 and 20, Fig. 13b), `--compressor topk`
(Section 5.6), `--track-workers` (per-worker traffic, Fig. 13).

Each run writes `results/<name>.json` with the config and per-round history
(simulated time, traffic, topology size, ratios, true and estimated consensus distance,
`D_max`, test accuracy and macro-F1 averaged over the workers' local models every
epoch, as in the paper, or every `--eval-every` rounds). It also stores each worker's label histogram.

## Implementation notes

**LP (27) is solved exactly without an LP library.** For a fixed topology the round time is
`max_i t_cp_i + r_i * beta_i` with `beta_i = B / min_{j in N_i} b_ij`, and the consensus
constraint reduces to `sum_j w_j r_j >= C`. The minimal feasible round time `t*` is found by
scanning the breakpoints of the piecewise-linear function `t -> sum_j w_j clip((t - t_cp_j)/beta_j)`.
The LP has many optimal solutions; `--tiebreak min_traffic` (default) picks the one with
the least traffic (fractional knapsack), and `--tiebreak max_ratio` gives every worker the
largest ratio that still fits within `t*`. The test checks `t*` against HiGHS on 200 random instances.

**Choices the paper leaves open:**

* `beta1 = 0.5`, `beta2 = 0.1` (EMA rates); `D_max` starts at the first measured update norm. `--dmax-scale` rescales it.
* Smallest compression ratio `--r-min 0.01` (the paper only requires `r > 0`).
* `D_ij` reported over a link is averaged over its two directions. The estimated average
  consensus distance that is logged is `sum_ij D_ij / m^2` (the upper bound of Eq. 22 with no communication).
* Dynamic bandwidth: each worker's in/outbound bandwidth follows a multiplicative random walk
  clipped to [1, 20] / [0.5, 10] Mb/s (`--volatility`). Computation time per round is drawn
  uniformly from [2.5, 7.5] s (VGG9) or [7.5, 12.5] s (ResNet9).
* Homogeneous network: every worker gets the mean of each range (10.5 Mb/s in, 5.25 Mb/s out).
* The learning rate decays once per epoch; weight decay is 1e-4; batch norm running statistics are averaged together with the weights.
* RandomK selects exactly `round(r*d)` indices from a seed shared with neighbours; TopK
  selects the largest-magnitude model entries. The extra cost of sending TopK indices is ignored, as in the paper.

**The baselines are simplified.** DCD, Choco-SGD and SAPS keep their topology and fixed-ratio
behaviour, but they use the same element-wise aggregation (Eq. 6) as CoCo. They do not
implement the original algorithms' error-compensated / difference-compression update rules, and
the SAPS peer selection is approximated by a max-bandwidth spanning tree.
