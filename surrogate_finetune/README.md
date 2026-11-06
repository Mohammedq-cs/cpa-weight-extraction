# Surrogates from a Partial Extraction: Labeled-Data Cost and PGD Transferability (Slurm)

**One-sentence summary:** Keep the first *K* CPA-recovered layers of a victim frozen, re-initialize
and fine-tune the rest on *N* labeled CIFAR-10 images, measure the smallest *N* that recovers the
victim's accuracy, and attack the victim with PGD adversarial examples crafted on those surrogates.
The figures are generated from `graphs_code/` (`postprocess.py` -> CSVs -> notebooks).

Covers three victims. The frozen prefix always comes from the **extracted** model (the output of
the sibling extraction pipelines), never from the original victim weights:

| Victim | Original model (attack target) | Original acc. | Extracted model (prefix source) | Extracted acc. as recovered | Layers |
|---|---|---|---|---|---|
| ResNet-18 | `../cpa_resnet/convNet.py`, `../cpa_resnet/resnet18.pt` | 92.59 | `models/recovered_resnet18.pt` (`../cpa_resnet`, 200k traces, HD) -- **partial extraction** | 10.38 | 21 |
| VGG-11-BN | `../vgg_e2e/convNet.py`, `../vgg_e2e/cifar10_vgg11_bn-eaeebf42.pt` | 92.78 | `models/recovered_vgg11bn.pt` (`../vgg_e2e`, 20k traces, HD) | 92.79 | 11 |
| small CNN | `../cpa_sim_multi_patch/convNet.py`, `../cpa_sim_multi_patch/model_cifar.pt` | 84.36 | `models/recovered_cnn8.pt` (`../cpa_sim_multi_patch`, 150k traces, HD) | 84.36 | 8 |

"Original acc." is the accuracy the surrogates must recover.
**ResNet-18 was not fully extracted:** every layer up to `layer4.1.conv1` is recovered at
0.96-1.0 bitwise accuracy, but `layer4.1.conv2` collapses to 0.43 and `fc` to 0.002, so the
extracted model as a whole scores 10.38% (`layer1.0.conv1`'s 0.84 is not a failure: its
mismatches sit on input positions that are always zero). Freezing *K* = 20 layers therefore
freezes the collapsed `layer4.1.conv2` as extracted and fine-tunes only `fc`. VGG-11 and the CNN
were extracted completely, so their extracted models match the originals.

The extracted weights are bundled in `models/recovered_{arch}.pt` (fp16 `{layer: weight}`), so
nothing here needs the extraction outputs. `build_recovered.py` rebuilds them from an extraction
run's per-filter pickles. Biases and BatchNorm are not extracted; the frozen prefix takes them
from the original model (see below).

---

## The experiment

Extraction is sequential (first layer -> last), so an attack that stops or degrades partway
leaves a recovered **prefix**. A cell (*K*, *N*) of the sweep:

- **keeps the first *K* conv/linear layers frozen** at their CPA-recovered weights, with the
  victim's bias for those layers (biases are not extracted) and their paired BatchNorm;
- **re-initializes the remaining layers** -- weight *and* bias -- with each architecture's own
  constructor init (ResNet/VGG: Kaiming-normal fan-out for convs; ResNet: PyTorch default for
  the fc layer; VGG: N(0, 0.01) for Linear; CNN: PyTorch defaults), and the BatchNorm paired with
  each of them (scale 1, shift 0, running stats reset), and trains all of those (`bnPrefix.py`).
  The running statistics of the frozen prefix BatchNorm recalibrate during training;
- trains on a **class-balanced subset of *N* labeled CIFAR-10 train images**,
  *N* in {10, 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 50k};
- **from scratch** (*K* = -1, plotted at x = depth): every layer *and* every BatchNorm random and
  trainable -- the "Surrogate" star.

x = "layers fine-tuned" = depth - *K*. Training: SGD (momentum 0.9, Nesterov), lr 0.05, weight
decay 5e-4, batch 128, cosine schedule, random-crop + flip, budget
`min(300 epochs, 5M samples seen)`, early stop 10 epochs after crossing victim - 1 std.
Evaluation is always the CIFAR-10 test split. Seed 8 draws both the labeled subset and the
initialization.

**Figure 1 -- `fig_data_vs_layers`:** the smallest *N* whose cell reaches test accuracy
>= victim - 1.1 at any evaluated epoch.

**Figure 2 -- `fig_transfer_vs_layers_archs_eps5255`:** for every depth, the surrogate trained on
**the same *N* that Figure 1 plots** (smallest *N* reaching victim - 1.1) is re-trained with its
model saved at the **first epoch** whose test accuracy reaches the cell's threshold. This is
repeated for training seeds **8, 9, 10** (seeds 9/10 re-draw the labeled subset and the
initialization; *N* is kept, and when a seed does not reach the threshold at that *N* its
highest-accuracy epoch is used). Each surrogate is the source of
an L-inf PGD attack on the victim over the CIFAR-10 test set: **eps = 5/255, alpha = 2/255,
75 steps, random start (attack seed 0), fp32 models + fp64 cross-entropy**.
ASR = victim error on the adversarial examples; each point is the **median over the three
training seeds**; the dashed line is the white-box attack (victim attacking itself).

---

## Repo entry points

- **Labeled-data sweep (Figure 1):** `run_sweep.sbatch` -> `sweepRepair.py`
- **Surrogate re-training with saved models (Figure 2):** `run_sources.sbatch` (reads `cells/transfer_sources.txt`) -> `sweepRepair.py --checkpoint_at_threshold`
- **PGD transfer (Figure 2):** `run_pgd.sbatch` -> `pgdSourceSweep.py` (+ `advTransfer.py`)
- **Architecture registry:** `archs.py` (paths, normalization, init, victim accuracies; `python archs.py` validates it)
- **Plots:** `graphs_code/postprocess.py` (results -> `graphs_code/data/*.csv`), then the notebooks in `graphs_code/` (CSV -> figure)

---

## Requirements

- Python 3.9.19, PyTorch / torchvision, pandas, matplotlib (single shared list at the repo root):
  ```bash
  pip install -r ../requirements.txt
  ```
- A GPU node (CUDA 12.2 assumed: `LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/`) for training and PGD.
- Slurm. Set `--partition` / `--nodelist` / `--constraint` in the three `run_*.sbatch` headers to
  match your cluster.
- CIFAR-10 from the repo-wide `../data/` folder, downloaded once for the whole repo — see
  [Datasets](../README.md#datasets) in the top-level README (`--data_root` overrides it).
- The sibling folders `../cpa_resnet`, `../vgg_e2e`, `../cpa_sim_multi_patch` (model definitions and
  victim checkpoints are loaded from there).

---

## Reproduce the figures

Run everything from this folder.

```bash
mkdir -p logs

# Figure 1: the K x N grid per architecture (12-shard arrays) -> results/{arch}/sweep_bnprefix/
for A in resnet18 vgg11bn cnn8; do sbatch --export=ALL,ARCH=$A run_sweep.sbatch; done

# Figure 2: re-train the 40 surrogates x seeds 8/9/10 (120 array tasks) -> results/{arch}/transfer_sources_seed{S}/
sbatch run_sources.sbatch
# ... then attack them (after the array has finished) -> results/{arch}/pgd_transfer_sources_eps5255_fp32_seed{S}/
for A in resnet18 vgg11bn cnn8; do for S in 8 9 10; do
  sbatch -J pgd_${A}_$S --export=ALL,ARCH=$A,SEED=$S run_pgd.sbatch; done; done

# Results -> CSVs -> figures (CPU only)
cd graphs_code
python postprocess.py            # writes data/fig_data_vs_layers.csv, data/fig_transfer_vs_layers_archs_eps5255.csv
jupyter nbconvert --to notebook --execute --inplace fig_data_vs_layers.ipynb
jupyter nbconvert --to notebook --execute --inplace fig_transfer_vs_layers_archs_eps5255.ipynb
# -> ../plots/fig_data_vs_layers.pdf, ../plots/fig_transfer_vs_layers_archs_eps5255.pdf
```

The CSVs produced from the paper's runs are committed in `graphs_code/data/`, so the notebooks
redraw the paper figures **without any results, model or GPU** -- skip straight to the two
`nbconvert` lines.

`cells/transfer_sources.txt` lists the attacked surrogates (`ARCH K N MODE THRESHOLD SEED`); its
*N* per depth is exactly Figure 1's (smallest size whose seed-8 sweep cell reached victim - 1.1),
and each surrogate is saved at an epoch with test accuracy >= victim - 1.1 (or, for a seed that
never gets there at that *N*, at its highest-accuracy epoch). The THRESHOLD column holds the exact
value each paper run checkpointed at.

### Single cell (no Slurm)

```bash
python sweepRepair.py --arch resnet18 --num_recovered 19 --train_size 1000 --out_dir results/resnet18/sweep_bnprefix
python sweepRepair.py --arch vgg11bn --num_recovered 8 --train_size 10 --seed 8 \
    --checkpoint_at_threshold 91.78 --out_dir results/vgg11bn/transfer_sources_seed8
python pgdSourceSweep.py --arch vgg11bn --ckpt_results_dir results/vgg11bn/transfer_sources_seed8 \
    --out_dir results/vgg11bn/pgd_transfer_sources_eps5255_fp32_seed8 \
    --eps 0.0196078431372549 --alpha 0.00784313725490196 --steps 75 --fp32 --attack_seeds 0
```

---

## Output layout

```
results/{arch}/sweep_bnprefix/K{K}_N{N}_cifar10_bnprefix.json             # one sweep cell: test-acc history
results/{arch}/transfer_sources_seed{S}/K{K}_N{N}_cifar10_bnprefix.json   # re-trained surrogate (+ checkpoint record)
results/{arch}/transfer_sources_seed{S}/checkpoints/{cell}_ep{E}.pt       # its saved model (fp32 state_dict)
results/{arch}/pgd_transfer_sources_eps5255_fp32_seed{S}/{cell}_ep{E}_seed0.json   # PGD stats per surrogate
results/{arch}/pgd_transfer_sources_eps5255_fp32_seed{S}/white_box_seed0.json
```

`results/` and `logs/` are not committed.
