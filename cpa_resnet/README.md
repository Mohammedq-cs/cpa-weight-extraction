# ResNet-18 Layer-by-Layer CPA Weight Extraction (Slurm)

**One-sentence summary:** Simulate recovering a pretrained **ResNet-18 (CIFAR-10)** model
layer-by-layer via **Correlation Power Analysis (CPA)** from synthetic Hamming-Weight (`HW`)
or Hamming-Distance (`HD`) leakage traces, using Slurm to fan out one job per filter.

The pipeline loads `resnet18.pt`, captures intermediate `Conv2d`/`Linear` activations on
CIFAR inputs, and recovers each filter's `float16` weights with CPA. Recovered weights are
written back into the model before the next layer is attacked, so extraction error compounds
across layers exactly as a real sequential attack would — final CIFAR-10 accuracy of the
fully substituted model is the headline metric.

1. **First-weight-known extraction.** The extractor is seeded with each filter's *true*
   first weight and recovers weights `1..N` via CPA on `prevSum` accumulators maintained in
   `float16`. This is an optimistic upper-bound assumption (it sidesteps the weight-0 sign
   ambiguity and the empty initial running sum); even so, extraction still degrades in the
   deeper ResNet layers. Removing/weakening this assumption is the open problem, not a
   feature to entrench — the principled fix is the reactive sign correction implemented in
   [`../vgg_e2e/`](../vgg_e2e/), which is not ported here.
2. **Per-weight multi-patch selection, decided over ALL traces on the GPU.** For each conv
   weight the pipeline picks the input patch whose column has the most distinct `float16`
   values (the strongest CPA target). That decision is made on the **full trace set**, not a
   subsample, using a CuPy distinct-counter (`gpuUniqueCount.py`) whose memory is independent
   of the trace count. The selection is derived from the **simulated** (attacker-side,
   recovered-so-far) activations, and the simulated *and* baseline patch memmaps for that
   selection are materialised together so they align 1:1.
3. **Disk-backed (memmap) inputs.** Each selected patch is stored as a column-major
   `(numWeights, numTraces)` `.npy` memmap, so a per-filter job streams only the columns it
   needs instead of holding every patch in RAM.
4. **Two leakage models, shared kernels.** `cpaCupy.py` provides three CuPy kernels —
   `hfm_kernel` (`float16` fused multiply-add), `popc_kernel` (Hamming weight), `hd_kernel`
   (XOR + popc). Trace math runs in `float16` to match a real attacker's observation
   precision; correlation accumulation upcasts to `float32`/`float64`.

> **Input normalization.** `resnet18.pt` was trained with the general CIFAR-10 channel
> statistics `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2023, 0.1994, 0.2010)` — **not**
> `Normalize(0.5)`. Measured CIFAR-10 test accuracy is **92.59%** with the correct stats vs
> **86.54%** under `Normalize(0.5)`; the mismatch corrupts the very activations the attack
> leaks from, so `demo.py` pins the correct values.

---

## Repo entry points

- **End-to-end run (CLI driver → Slurm):** `submit_experiment.sh` → `demo.py`
  (ready-made single run: `run200k.sbatch`)
- **Per-filter extractor (one Slurm array task, generated at runtime):**
  `extractFilterFirstWeightKnownMultiPatchesOptimized.py`
- **Cluster config:** `slurm_config.py`

Shared modules:

| File | Role |
|---|---|
| `simulation.py` | per-layer orchestration: patch selection, sim + baseline memmap capture, unique-count cache, Slurm fan-out, recovered-vs-ground-truth evaluation, weight propagation |
| `patchInputs.py` | patch selection, memmap layout/IO, window buffering, selection-mismatch flagging, unique-count cache persistence |
| `gpuUniqueCount.py` | CuPy all-trace distinct-value counter (packed presence bitset + `__popc`); byte-identical to the CPU counter, ~50–200x faster |
| `uniqueCount.py` | the CPU distinct-value counter (fallback + reference) |
| `cpaExtractCore.py` | CPA primitives shared by the extractor (`IncrementalLeakage`, per-weight unique-count matrix) |
| `slurmManager.py` | generates + submits the per-filter array jobs from `slurm_config.py` |
| `attack.py` | leakage-hypothesis / `prevSum` logic |
| `cpaCupy.py` | GPU kernels |
| `convNet.py` | the CIFAR ResNet-18 (plus a small CNN, unused here) |
| `modelUtils.py` | activation hooks, weight updates |
| `datasets.py` | CIFAR-5M loader |
| `utils.py` | accuracy, weight candidates, bitwise-accuracy reporting |

### Layers attacked

In `named_modules()` order, the `Conv2d`/`Linear` layers the pipeline walks are:

```
conv1
layer1.0.conv1, layer1.0.conv2,                        layer1.1.conv1, layer1.1.conv2
layer2.0.conv1, layer2.0.conv2, layer2.0.downsample.0, layer2.1.conv1, layer2.1.conv2
layer3.0.conv1, layer3.0.conv2, layer3.0.downsample.0, layer3.1.conv1, layer3.1.conv2
layer4.0.conv1, layer4.0.conv2, layer4.0.downsample.0, layer4.1.conv1, layer4.1.conv2
fc
```

`layerN.0.downsample.0` is the 1x1 conv inside the residual `downsample` Sequential; it
shares the block's input activation with `conv1` but gets its own patch selection.
BatchNorm / ReLU / pooling layers are skipped automatically.

---

## Requirements

- Python 3.9.19
- GPU node(s) with CUDA (this repo assumes CUDA 12.2 by default)
- **CuPy** (CUDA 12.x), **cuda-python**, PyTorch, torchvision, NumPy, pandas, tqdm, ipdb
- Slurm access
- Enough disk for the per-layer patch memmaps + run outputs

Install the shared dependency list at the repo root:
```bash
pip install -r ../requirements.txt
```

GPU-only. The kernels require `LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/`.

> **GPU memory for patch selection.** `gpu_unique_count_matrix` blocks its presence bitset to
> a `mem_budget_bytes=1_500_000_000` default so it fits 12 GB-class cards; raise it if your
> GPUs are larger.

---

## Configure Slurm (`slurm_config.py`)

Edit `slurm_config.py` to match your cluster — `slurmManager.py` reads it to generate every
per-filter array job:

```python
job_name_suffix = "Exper"
output_file = "./scriptsOut/filter"
error_file  = "./scriptsErr/filter"
time_limit  = "1400"       # minutes or Slurm format D-HH:MM:SS
nodes = 1
ntasks = 1
memory = 130000            # MB (per-filter peak ~80GB; the higher reservation
                           # discourages co-locating two filter jobs on one node)
cpus_per_task = 1
gpus = 1
nodelist = ""              # set to your cluster's node(s), or leave empty
array_concurrency = 32     # throttle concurrent per-filter array jobs
partition = ""             # set to your cluster's partition, or leave empty
constraint = "geforce_rtx_3090"
LD_LIBRARY_PATH = "/usr/local/cuda-12.2/lib64/"
```

> `nodelist`, `partition`, and `constraint` must match your cluster naming. The
> `--nodelist`/`--partition`/`--constraint` lines in `submit_experiment.sh` and
> `run200k.sbatch` (the outer orchestrator job) are commented placeholders — fill them in
> the same way.

---

## Datasets & model

- Datasets: CIFAR-10 and CIFAR-5M are downloaded **once** for the whole repo — see
  [Datasets](../README.md#datasets) in the top-level README. The scripts read them from `../data`. `demo.py` reads `part0.npz`.
- **Pre-trained ResNet-18 weights** are bundled in this folder: `resnet18.pt`
  (loaded by `demo.py`, half precision).

---

## Run end-to-end on Slurm

Create log folders (recommended):
```bash
mkdir -p scriptsOut scriptsErr
```

**Ready-made 200k run.** A self-contained job for a single 200k-trace extraction is bundled
as `run200k.sbatch` (HD, noise 0, seed 8, `cifar6m`):
```bash
mkdir -p ./N0/S8/resnet_fwk/cifar6m/HD/200k
sbatch run200k.sbatch
```

Submit through the CLI driver (generates `temp_{N}_{k}_{LEAKAGE}.sbatch` and submits it):
```bash
./submit_experiment.sh -e <ExperName> -l <HW|HD> -n <NOISE_LEVEL> \
    -k <kTraces> -d <cifar10|cifar6m|random> [-s <script_basename>] \
    [-r <seed>] [-j <dependency_job_id>]
# e.g. ./submit_experiment.sh -e resnet_fwk -l HD -n 0 -k 200 -d cifar6m -r 8
```
`-k` is in thousands of traces (`-k 200` -> 200000). `-s` is the Python script basename
without `.py` (default `demo`). Outputs land under
`./N{noise}/S{seed}/{ExperName}/{dataset}/{leakage}/{k}k/`.

…or run the orchestrator directly on a GPU node:
```bash
python demo.py \
    --model_save_path ./extracted_model.pth \
    --num_traces 200000 --noise_level 0.0 --leakage_model HD \
    --exper_name resnet_fwk --dataset_name cifar6m --seed 8
```

`demo.py` loads `resnet18.pt` as the frozen ground-truth `true_model`, saves its per-filter
weights as reference (`./model_weights/{layer}/filter_{i}.pkl`), makes a zeroed
`working_model` copy, then runs `runWeightsRetrievalSimulationFirstWeightKnown`
(`simulation.py`). There is **no up-front baseline generation**; per layer it:

1. captures the simulated activation from `working_model` (and the baseline activation from
   `true_model`) via hooks,
2. computes the per-weight patch selection from the simulated activation over **all** traces
   on the GPU,
3. writes the simulated **and** baseline patch memmaps for that selection together,
4. caches the per-(patch, weight) unique-count matrix as `{layer}.unique_counts.pkl`,
5. fans out a Slurm array of `extractFilterFirstWeightKnownMultiPatchesOptimized.py`
   (one task per filter),
6. loads the recovered filters, prints bitwise accuracy + a weight-difference histogram,
7. propagates the recovered weights into the live model and deletes that layer's memmaps.

Finally it saves the substituted model and evaluates CIFAR-10 accuracy.

### Useful `demo.py` flags

- `--start_at_layer <layerName>` — resume: skip extraction for every layer before this one,
  loading their already-recovered per-filter pickles into the model instead. Use this to
  restart a run from the layer that was in flight when it died.
- `--keep_layer_memmaps` — keep each layer's transient sim + baseline patch memmaps instead
  of deleting them after the layer completes. Debug only: they are tens of GB per layer and
  will exhaust a disk quota over a full run.

### Single filter (no Slurm)
```bash
python extractFilterFirstWeightKnownMultiPatchesOptimized.py \
    --baseLineInputDir <.../baseLineInps/{exper}/{dataset}/{numTraces}> \
    --simulatedInputsDir <.../simInps/N{noise}/seed{S}/{exper}/{leakage}/{dataset}/{numTraces}> \
    --corrStatSavePath <...> --outputDir <...> \
    --layerName layer4.1.conv2 --filterIdx 0 --NoiseLevel 0.0 --leakage_model HD
```
The job resumes from `{outputDir}/{layerName}/filter_{i}_checkpoint.pkl` if present. The
checkpoint carries `prevSum_list`, the incremental-leakage accumulators, and the
chosen-patch / unique-count history — do not change its structure without updating the
resume logic.

---

## Output layout

With `exper`/`dataset`/`leakage`/`seed` as passed on the command line:

```
./model_weights/{layer}/filter_{i}.pkl                         ground-truth reference weights
./baseLineInps/{exper}/{dataset}/{numTraces}/{layer}.patch{p}.npy
                                                               true-model patch memmaps
./simInps/N{noise}/seed{S}/{exper}/{leakage}/{dataset}/{numTraces}/{layer}.patch{p}.npy
                                                               working-model patch memmaps
./simInps/.../{numTraces}/{layer}.unique_counts.pkl            per-layer unique-count cache
./N{noise}/seed{S}/{exper}/{leakage}/{dataset}output/{numTraces}/{layer}/filter_{i}.pkl
                                                               recovered filter weights
./N{noise}/seed{S}/{exper}/{leakage}/{dataset}Corr/{numTraces}/{layer}/filter_{i}.pkl
./N{noise}/.../{layer}/filter_{i}_chosen_patches.pkl           per-filter correlation stats
./scriptsOut/filter_{i}.out, ./scriptsErr/filter_{i}.err       per-filter Slurm logs
{--model_save_path}                                            final substituted model (.pth)
```

The patch memmaps are transient scratch (deleted per layer unless `--keep_layer_memmaps`).
Run outputs, Slurm logs, and dataset caches are not committed.

---

## Generate graphs (paper figure)

Per-layer bitwise accuracy of the 200k run (`run200k.sbatch`), split into two steps so the
figure can be redrawn without the raw results:

1. **Post-process** (CPU; reads `model_weights/` and the recovered `filter_*.pkl`) -> `graphs_code/data/resnet_bitwise_acc.csv`:
   ```bash
   cd graphs_code
   python postprocess.py      # --root <results> --run <recovered-filter dir of the run>
   ```
2. **Plot**: open `graphs_code/resnet_e2e_bitwise_acc_squashed.ipynb` and **Run All**. It only reads
   the CSV and writes `../plots/resnet_bitwise_acc.pdf` and `../plots/paper/resnet_bitwise_acc_squashed.pdf`.

The CSV used for the paper is committed, so step 2 works on a fresh clone.

---

## Troubleshooting

- **Jobs fail immediately:** check `scriptsErr/*` for missing imports or dataset paths.
- **CUDA errors:** verify `LD_LIBRARY_PATH` and the node's CUDA/driver, and that CuPy matches
  the CUDA version.
- **`[selection] GPU all-trace selection unavailable …`:** CuPy/CUDA could not be reached, so
  patch selection silently fell back to the CPU counter over the first 20,000 traces. Results
  are still valid but the selection is no longer decided over the full trace set — fix the
  GPU environment if you need the all-trace behaviour.
- **`OSError: Disk quota exceeded` mid-run:** you are almost certainly running with
  `--keep_layer_memmaps`. Drop it.
- **Dataset not found:** confirm `../data/cifar-10-batches-py/` and `../data/cifar5m/part*.npz` exist
  (or that `CPA_DATA` points to a folder containing them).
- **Hooks capture the wrong activations:** `modelUtils.intermediateActivations` is a global
  dict keyed by module identity, so the orchestrator can hook both models at once — but it
  owns setup/teardown for both. Remove hooks before re-running anything by hand.
