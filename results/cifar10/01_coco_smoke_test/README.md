# CIFAR-10 — first CoCo run

Single CoCo run on CIFAR-10 with VGG9, partition "dominant" with p = 10 (= IID in the paper's scheme),
200 epochs, `heter_dynamic` network, seed 0. Used to check the implementation.

```bash
python main.py --algo coco --dataset cifar10
python plot.py results/cifar10/01_coco_smoke_test/*.json --target 0.87 -o fig_cifar10.png
```

| Figure | Content |
|---|---|
| `fig_cifar10.png`, `fig_results.png` | `plot.py` 3×3 grid (accuracy, loss, consensus, ratios, links) |
| `fig_cifar10_ratios.png` | compression ratio per worker × round |
