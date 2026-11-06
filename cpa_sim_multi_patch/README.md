# Multi-Patch CPA Weight Extraction: CIFAR-10 + CIFAR-5M (single-node, in-process)

**One-sentence summary:** Run the full Correlation-Power-Analysis (CPA) weight-recovery
pipeline with **per-weight multi-patch selection** on **CIFAR-10** and **CIFAR-5M** as a
single **in-process** job (one SLURM allocation, one or more GPUs), then generate the paper
graphs from `graphs_code/`.

This variant keeps the per-weight multi-patch CPA target selection — for every conv weight
it picks the kernel-application patch whose input column has the most distinct `float16`
values (the strongest CPA target), and the per-layer set is the union of those argmax
patches — but changes **how the run is executed and how activations are fed**:

- **In-process, single-node execution.** `demo.py` runs the whole extraction inside one
  process. Per-filter extraction runs either inline on a single GPU (`InlineExecutor`) or
  across the node's GPUs via a worker pool (`PoolExecutor`, `--num_gpus N`); no per-filter
  SLURM jobs.
- **Patch selection from the simulated activations, over all traces.** For each layer the
  per-weight best patch is decided from the `working_model` (attacker-side) activation over
  **all** traces on the GPU (`gpuUniqueCount.py`), and the per-(patch, weight) unique-count
  matrix used by the extractor is computed on the GPU as well.
- **Activations held in RAM.** Both the frozen `true_model` and the zeroed `working_model`
  are held in memory; their activations are captured and served column by column on demand
  by `columnProvider.ColumnProvider`. Only the per-filter recovered weights and correlations
  are written to disk (see *Outputs* below).
- **Per-filter resume.** Filters already saved under the output folder are reused when a run
  is restarted, so an interrupted layer continues where it stopped.
- **Last-layer sign fix.** After extraction, `lastLayerFix.correct_last_layer_signs` resolves
  the sign ambiguity on the final `fc2` layer (which has no BatchNorm to pin the sign).

`test_column_parity.py` (below) checks that the column serving is byte-identical to
`F.unfold`.

---

## Repo entry points

- **Submit one experiment:** `submit_experiment.sh` (flags below).
- **Zero-noise sweep:** `run_zero_noise.sh`
- **Noise sweep (runs for the paper figure):** `run_expers_with_noise.sh`
- **Random-models sweep:** `run_expers_rnd_models.sh`
- **Cluster config:** the `#SBATCH` block is written inline by the `submit_experiment*.sh` wrappers.
- **Plots:** `graphs_code/postprocess.py` (results -> `graphs_code/data/*.csv`), then the notebooks in `graphs_code/` (CSV -> figure)

Python entry scripts dispatched by the run scripts (all run on the in-process engine):
- `demo.py` — main multi-patch weight-extraction run (`-s demo`, the default)
- `inpDist.py` — input-distribution comparison across datasets (`-s inpDist`)
- `model_choice.py` — pretrained / random-init / random-uniform model variants
  (`-s model_choice`, selected with `--model`)

Correctness test (no SLURM needed):
- `test_column_parity.py` — CPU-only, fast: asserts `ColumnProvider.column(p,w)` is
  byte-identical to `F.unfold(A)[:, w, p]` across conv configs (incl. padded edges) + Linear.

---

## Requirements

- Python 3.9.19
- GPU node(s) with CUDA (this repo assumes CUDA 12.2 by default)
- SLURM access (one single-node job per run)
- Enough disk for datasets + per-filter outputs

Install the shared dependency list at the repo root:
```bash
pip install -r ../requirements.txt
```

---

## Datasets

Datasets: CIFAR-10 and CIFAR-5M are downloaded **once** for the whole repo — see
[Datasets](../README.md#datasets) in the top-level README. The scripts read them from `../data`.

### Model files (included)
The reference model the extractor loads is bundled at the repo root: `model_cifar.pt`.
`model_choice.py` additionally loads the random models `model_rand_init.pth` (`--model randInit`)
and `model_rand_uniform.pth` (`--model randUniform`); `inpDist.py` loads the accuracy-bucket
indices in `buckets_pickles/` for the `c6m_low_acc` / `c6m_high_acc` datasets.

---

## Run end-to-end on SLURM

Create log folders (recommended):
```bash
mkdir -p scriptsOut scriptsErr
```

### `submit_experiment.sh` flags
```
-e ExperName   -l LEAKAGE_MODEL (HW|HD)   -n NOISE_LEVEL   -k kTraces (in thousands)
-d Dataset_name   [-s ScriptName=demo]   [-r seed=8]   [-j dependency_job_id]
```
It writes a single-GPU sbatch job that runs:
```
python demo.py --model_save_path ... --num_traces ... --noise_level ... \
               --leakage_model ... --exper_name ... --dataset_name ... --seed ...
```
Example:
```bash
./submit_experiment.sh -e model_extract -l HD -n 0 -k 10 -d cifar6m -s demo -r 8
```

### `demo.py` flags
```
--model_save_path PATH   (required)   where the recovered .pth is saved
--num_traces N           (required)   number of CIFAR-5M traces to use
--noise_level F          default 0.0
--leakage_model HW|HD    default HW
--exper_name NAME        default accuracy_experiment
--dataset_name {cifar10,cifar6m,random}   default cifar6m
--seed N                 default 8
--num_gpus N             default 1   >1 -> PoolExecutor across the node's GPUs
--output_dir DIR         default ./N{noise}/{exp}output/{numTraces}
--layers_subset a,b,...  restrict to these layer names (debug)
```
> The bundled sbatch requests `--gpus=1`, so the default `--num_gpus 1` (inline, single GPU)
> is used. For multi-GPU, raise `--gpus` in `submit_experiment.sh` and pass `--num_gpus N`.

### 1) Zero-noise sweep
Chains the trace-count (and input-distribution) experiments:
```bash
./run_zero_noise.sh                # start fresh
./run_zero_noise.sh <dep_job_id>   # chain after an existing job (optional dependency)
```
The sweep chains a `model_extract` (`-s demo`) phase across trace counts and an
`inp_dis_comp` (`-s inpDist`) phase across datasets — both on the in-process engine.

### 2) Random-models sweep
```bash
./run_expers_rnd_models.sh
```
Runs `model_choice.py` (`-s model_choice`) on the random-init / random-uniform variants.
`verify_then_launch_rnd_models.sh` checks the zero-noise `.pth` outputs are all present
before kicking this off.

## Outputs

For each run, `demo.py` writes (with `exp = seed{seed}/{exper_name}/{leakage_model}/{dataset_name}`):
```
./N{noise}/{exp}output/{numTraces}/{layer}/filter_{idx}.pkl   # recovered per-filter weights
./N{noise}/{exp}Corr/{numTraces}/{layer}/...                  # per-filter correlations
./scriptsOut/{exp}/{numTraces}/...                            # per-filter extractor chatter
{model_save_path}                                            # final recovered model state_dict
```
The model written to `{model_save_path}` is the **sign-corrected** model: `lastLayerFix.correct_last_layer_signs`
has already been applied to resolve the `fc2` sign ambiguity, so the saved `state_dict` is the
final, ready-to-evaluate recovered model (not the raw sign-ambiguous extraction output).

Keep this folder structure stable so `graphs_code/postprocess.py` can find the results.

---

## Generate graphs (paper figures)

Graph generation is split into two steps, so the figures can be redrawn without the raw results:

1. **Post-process** (GPU; reads the experiment outputs) -> tidy CSVs in `graphs_code/data/`:
   ```bash
   cd graphs_code
   python postprocess.py        # --root <results> / --first-patch-root <cpa_sim_first_patch results>
   ```
   The figure compares against the single-RF runs of `../cpa_sim_first_patch` (default
   `--first-patch-root`).
2. **Plot**: open `noise_ext_comp.ipynb` and **Run All**. It only reads `graphs_code/data/noise_ext_comp.csv`
   and writes `./plots/single_vs_multi_noise.pdf` (accuracy vs σ, Adaptive RF vs Fixed RF).

The CSV used for the paper is committed, so step 2 works on a fresh clone.
Recomputing them can change the model accuracies slightly (~1e-4, i.e. one test image)
depending on the GPU architecture (fp16 cuDNN kernels).
`utilsFuncs.py` holds the shared helpers used by `postprocess.py`.

---

## Troubleshooting

- **Job fails immediately:** check the job's `exper.err` for missing imports or dataset paths.
- **CUDA errors:** verify `LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64/` and that the node has
  the expected CUDA/driver.
- **Dataset not found:** confirm `../data/cifar-10-batches-py/` and `../data/cifar5m/part0.npz` exist
  (or that `CPA_DATA` points to a folder containing them).
- **Out of memory:** lower `--num_traces`, or restrict layers with `--layers_subset` while
  debugging. With `--num_gpus > 1` the per-filter work is spread across GPUs.
