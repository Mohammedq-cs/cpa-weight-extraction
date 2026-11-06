# VGG-11 End-to-End Sign-Ambiguous CPA Weight Extraction (Slurm)

**One-sentence summary:** Recover a full **VGG-11 (BN)** model on **CIFAR-10 / CIFAR-5M**
end-to-end via **sign-ambiguous Correlation Power Analysis (CPA)** followed by **reactive
sign correction**, using Slurm to fan out one job per filter.

Every Conv2d/Linear layer is first recovered up to a global ± sign with multi-patch CPA,
then sign-corrected using the *next* layer as an oracle. The pipeline reuses the same
multi-patch + memmap machinery as the CPA simulation repos:

1. **Per-weight patch selection.** For each conv weight the pipeline materialises the
   kernel-application patch whose input column has the most distinct `float16` values
   (the strongest CPA target). The per-layer saved set is the union of those argmax patches,
   computed once from the real model and persisted, so baseline and simulated patch lists
   align 1:1.
2. **Disk-backed memmap IO.** Each selected patch is stored transposed as a
   `(numWeights, num_traces)` `.npy` memmap; the CPA loop streams columns in small windows
   and the true-filter leakage trace is accumulated incrementally, so RAM stays flat
   (~60 GB) regardless of trace count.
3. **Sign correction.** Conv2d oracles use a reactive head-to-head walk; Linear oracles use
   a head-to-head sweep with a majority vote; the terminal classifier layer is fixed using
   per-class test-set accuracy.

---

## Repo entry points

- **End-to-end run (orchestrator):** `runSignAmbiguous.sbatch` → `demoSignAmbiguous.py`
- **Per-filter extractor (one Slurm array task):**
  `extractFilterSignAmbiguousMultiPatchesOptimized.py`
- **Cluster config:** `slurm_config.py`

Shared modules: `patchInputs.py` (layer-input extraction + patch selection + memmap IO),
`cpaExtractCore.py` (CPA primitives), `uniqueCount.py` (fast `float16` distinct-value
counter), `signCorrection.py` (the three sign-correction strategies), `simulation.py`
(per-layer orchestration).

---

## Requirements

- Python 3.9.19
- GPU node(s) with CUDA (this repo assumes CUDA 12.2 by default)
- **CuPy** (CUDA 12.x), PyTorch, torchvision, NumPy, pandas
- Slurm access
- Enough disk for the per-patch input memmaps + run outputs

Install the shared dependency list at the repo root:
```bash
pip install -r ../requirements.txt
```

GPU-only. The kernels require `LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/`.

---

## Configure Slurm (`slurm_config.py`)

Edit `slurm_config.py` to match your cluster:

```python
job_name_suffix = "Exper"
output_file = "./scriptsOut/filter"
error_file  = "./scriptsErr/filter"
time_limit  = "1400"       # minutes or Slurm format D-HH:MM:SS
nodes = 1
ntasks = 1
memory = 60000             # MB
cpus_per_task = 1
gpus = 1
nodelist = ""              # set to your cluster's node(s), or leave empty
array_concurrency = 32     # throttle concurrent per-filter array jobs
partition = ""             # set to your cluster's partition, or leave empty
constraint = "geforce_rtx_3090"
LD_LIBRARY_PATH = "/usr/local/cuda-12.2/lib64/"
```

> `nodelist`, `partition`, and `constraint` must match your cluster naming.

---

## Datasets & model

- Datasets: CIFAR-10 and CIFAR-5M are downloaded **once** for the whole repo — see
  [Datasets](../README.md#datasets) in the top-level README. The scripts read them from `../data`.
- **Pre-trained VGG-11 (BN) weights** are bundled at the repo root:
  `cifar10_vgg11_bn-eaeebf42.pt` (loaded by `demoSignAmbiguous.py`).

---

## Run end-to-end on Slurm

Create log folders (recommended):
```bash
mkdir -p scriptsOut scriptsErr
```

Submit the orchestrator:
```bash
sbatch runSignAmbiguous.sbatch
```

…or run it directly on a GPU node:
```bash
python demoSignAmbiguous.py \
    --model_save_path ./extracted_model.pt \
    --num_traces 40000 --noise_level 0.0 --leakage_model HD \
    --exper_name vgg_e2e --dataset_name cifar6m --seed 8
    # --g_idx 0            (Conv2d-oracle filter index)
    # --fc_num_probes 8    (Linear-oracle voting probes; last conv + classifiers)
    # --save_unique_counts (diagnostic: per layer, dump per-weight distinct-value
    #                       counts for the simulated + baseline inputs over the saved
    #                       patches to ./uniqueCounts/N{noise}/{exper_name}/{numTraces}/)
```

> Sign correction has **no detection-threshold knob**: the Conv2d and Linear paths
> both run an *unconditional* head-to-head sweep over every upstream filter and
> commit a flip on a relative majority vote. (The earlier reactive-trigger flags —
> `--corr_threshold`, `--M`, `--no_sanity_verify`, `--verify_threshold` — were
> dead in this pipeline and have been removed.)

`demoSignAmbiguous.py` writes the baseline memmaps + selection from the real model, zeros
the model, then runs `runWeightsRetrievalSimulationSignAmbiguous` (`simulation.py`), which
per layer: generates simulated-input memmaps (reusing the saved selection — and if any
weight's *simulated* best patch falls outside that real-derived selection, it appends the
missing patch to the baseline memmaps and grows `{layer}.selection.npy` so the union stays
1:1-aligned, rather than falling back), fans out a
Slurm array of `extractFilterSignAmbiguousMultiPatchesOptimized.py` jobs (one per filter),
loads the ambiguous filters, runs sign correction against the downstream oracle, and
evaluates.

### Trace-count sweep

To run the standard sweep — **500, 1k, 2k, 5k, 10k, 20k, 40k** traces — there are two launchers.
Both submit one orchestrator job per trace count, write each run's logs and model under
`./N0/S8/vgg_e2e/cifar6m/HD/<label>/` and its recovered filters under
`./N0.0/seed8/vgg_e2e/HD/cifar6moutput/<numTraces>/` (the layout
`graphs_code/postprocess.py` reads back), and accept an optional custom
list of trace counts as arguments.

```bash
./launchTraceSweep.sh            # all six runs submitted at once (parallel)
./launchTraceSweepChained.sh     # the six runs CHAINED, one after another
./launchTraceSweepChained.sh 500 1000 5000   # custom subset, still chained
```

### Graph

```bash
cd graphs_code
python postprocess.py      # recovered filters vs model_weights -> data/bitwise_acc.csv, data/crash_layer_by_traces.csv
```
then open `bitwise_acc_up_to_failure.ipynb` and **Run All**; it only reads `data/*.csv` (committed)
and writes `../plots/bitwise_acc_up_to_failure.pdf`.

`launchTraceSweepChained.sh` submits each trace count with
`--dependency=afterany:<previous job id>`, so trace count *N+1* starts only after trace
count *N* finishes (success **or** failure — `afterany`, matching `run_exps.sh` elsewhere in
the repo). Use it when you want the sweep to share the cluster politely (one orchestrator +
its per-filter array at a time) instead of contending for GPUs in parallel; use the plain
`launchTraceSweep.sh` when you have the capacity to run them all concurrently.

### Single filter (no Slurm)
```bash
python extractFilterSignAmbiguousMultiPatchesOptimized.py \
    --baseLineInputDir <.../baseLineInps/.../{numTraces}> \
    --simulatedInputsDir <.../simInps/.../{numTraces}> \
    --corrStatSavePath <...> --outputDir <...> \
    --layerName features.4 --filterIdx 0 --NoiseLevel 0.0 --leakage_model HD
```

---

## Troubleshooting

- **Jobs fail immediately:** check `scriptsErr/*` for missing imports or dataset paths.
- **CUDA errors:** verify `LD_LIBRARY_PATH` and the node's CUDA/driver, and that CuPy matches
  the CUDA version.
- **Dataset not found:** confirm `../data/cifar-10-batches-py/` and `../data/cifar5m/part*.npz` exist
  (or that `CPA_DATA` points to a folder containing them).
- **Wrong-sign filters in a layer:** the head-to-head majority rule — not an absolute
  correlation gate — is the robust decision for the sign of each filter. The wrong-sign CPA
  "floor" is non-trivial (`BN(−z) = −BN(z) + 2b` reflected through ReLU), so an absolute
  threshold can't separate it from a true peak; that is why the unconditional relative sweep
  replaces any detection trigger.
