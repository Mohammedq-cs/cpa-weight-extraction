# End-to-End Repro: CIFAR-10 + CIFAR-5M (Slurm)

**One-sentence summary:** Run the full pipeline on **CIFAR-10** and **CIFAR-5M** using Slurm via `run_exps.sh` and `run_expers_with_noise.sh`, then generate the paper graphs from `graphs_code/` (`postprocess.py` -> CSVs -> notebooks).

---

## Repo entry points

- **Main Slurm sweep (no noise):** `run_exps.sh`
- **Noise sweep:** `run_expers_with_noise.sh`
- **Cluster config:** `slurm_config.py`
- **Plots:** `graphs_code/postprocess.py` (results -> `graphs_code/data/*.csv`), then the notebooks in `graphs_code/` (CSV -> figure)

---

## Requirements

- Python 3.9.19
- GPU node(s) with CUDA (this repo assumes CUDA 12.2 by default)
- Slurm access
- Enough disk for datasets + run outputs

Install Python dependencies (single shared list at the repo root):
```bash
pip install -r ../requirements.txt
```

---

## Configure Slurm (`slurm_config.py`)

Edit `slurm_config.py` to match your cluster:

```python
# slurm_config.py

# ---- SLURM settings ----
job_name_suffix = "CPA Simulation"
output_file = "./scriptsOut/filter"
error_file  = "./scriptsErr/filter"
time_limit  = "1400"       # minutes or Slurm format D-HH:MM:SS (depending on your submitter)
nodes = 1
ntasks = 1
memory = 60000            # MB
cpus_per_task = 1
gpus = 1
nodelist = ""
array_concurrency = 32
partition  = ""
constraint = "geforce_rtx_3090"

# ---- Environment exports ----
LD_LIBRARY_PATH = "/usr/local/cuda-12.2/lib64/"
```

> Notes  
> - `nodelist`, `partition`, and `constraint` must match your cluster naming.  
> - `array_concurrency` is relevant only if your scripts use Slurm job arrays.

---

## Datasets

Datasets: CIFAR-10 and CIFAR-5M are downloaded **once** for the whole repo — see
[Datasets](../README.md#datasets) in the top-level README. The scripts read them from `../data`.

---

## Run end-to-end on Slurm

Create log folders (recommended):
```bash
mkdir -p scriptsOut scriptsErr
```

### 1) Main sweep (no noise)
Submits the main experiment sweep (trace counts × input distributions, etc.):
```bash
bash run_exps.sh
```

### 2) Noise sweep
Submits the noisy experiment sweep:
```bash
bash run_expers_with_noise.sh
```

### 3) Random models sweep
Submits the Random models experiements sweep:
```bash
bash run_expers_rnd_models.sh
```

Slurm logs are written under:
- `./scriptsOut/…`
- `./scriptsErr/…`

Keep the output folder structure stable so `graphs_code/postprocess.py` can find the results.

---

## Generate graphs (paper figures)

Graph generation is split into two steps, so the figures can be redrawn without the raw results:

1. **Post-process** (GPU; reads the experiment outputs) -> tidy CSVs in `graphs_code/data/`:
   ```bash
   cd graphs_code
   python postprocess.py                         # all figures
   python postprocess.py --only zero_noise_graph # just one notebook's data
   python postprocess.py --root /path/to/results # results stored elsewhere (same layout as ./)
   ```
2. **Plot**: open each notebook in `graphs_code/` (one per paper figure) and **Run All**. The
   notebooks only read `graphs_code/data/*.csv` and write the PDFs to `./plots/`.

The CSVs used for the paper are committed, so step 2 works on a fresh clone.
Recomputing them can give slightly different numbers depending on the GPU architecture (cuDNN
kernels): model accuracies and PGD success rates by ~1e-4 (one or two test images), up to ~1e-2
for the near-chance models extracted from few traces, and `inp_errs_probs` error rates by up to a
few 1e-2.
`err_collision` additionally needs the (large, ~100 GB) `preprocess_collision.py` checkpoints in
`graphs_code/chkpts/` (`--chkpts` to point elsewhere); they are created on first use.

---

## Troubleshooting

- **Jobs fail immediately:** check `scriptsErr/*` for missing imports or missing dataset paths.
- **CUDA errors:** verify `LD_LIBRARY_PATH` and that the node has the expected CUDA/driver.
- **Dataset not found:** confirm `../data/cifar-10-batches-py/` and `../data/cifar5m/part*.npz` exist
  (or that `CPA_DATA` points to a folder containing them).
- **Too many jobs running:** reduce `array_concurrency` in `slurm_config.py` or limit jobs inside the run scripts.
