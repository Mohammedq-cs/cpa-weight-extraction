# Extraction with a Realistic Signal-to-Noise Ratio (Section 8.1)

**One-sentence summary:** Reproduce the Section 8.1 results — CPA weight recovery of
EfficientNet-B0 and a small CIFAR CNN at the noise level extrapolated from prior physical
measurements (σ = 850) over **12.24 M traces** — from scratch, including regenerating every
activation trace.

This folder covers the noisy, large-trace regime. Two attack modes are compared at each
noise level:

- **Sequential** — each weight's hypothetical partial sum is advanced with the
  **previously recovered** weights, so recovery errors propagate. This is the realistic
  attacker.
- **GT-oracle** (`--prev_known`) — the partial sum is advanced with the **ground-truth**
  previous weights, suppressing error propagation. The gap between the two isolates how much
  of the failure is *propagation* rather than *leakage*.

Both use the **adaptive receptive field**: for every weight, the kernel-application patch
whose input column carries the most distinct `float16` values is selected as the CPA target.

## What it reproduces

| Target | Model / inputs | Expected |
|---|---|---|
| Figure 19 sweep, sequential | EfficientNet-B0 `features.0.0`, ImageNet | 1.0000, .9965, .9479, .9225, .8542, .7569, .5590, .4583, **.3391** for σ = 0…850 |
| Figure 19 sweep, GT-oracle | same | 1.0000, .9965, .9780, .9537, .9144, .8866, .8507, .8171, **.7280** |
| Layer-2 chained, σ=850 | `features.1.0.block.0.0`, hypotheses from *extracted* layer-1 weights | 5/288 = **.0174** |
| Random-model calibration, σ=850 | uniform(−5,5) weights, uniform[−2,2] inputs, oracle | L1 741/864 = **.8576** |
| CIFAR CNN `conv1`, σ=850 | CIFAR-5M, sequential | 331/864 = **.3831** |

> **Erratum — random-model layer 2.** The paper reports 80.56% for the depthwise layer of the
> random model. That figure was produced with a baseline that stored only **channel 0's**
> 3×3 patch and reused it for all 32 filters. In a depthwise convolution
> (`groups == in_channels`) filter *f* convolves **only channel *f***, so the attack was run
> against inputs the model never uses. The attack was internally consistent — the true
> leakage and the hypotheses came from the same wrong columns — which is why it produced a
> plausible number rather than failing. With the channel-correct baseline in this code
> (all `in_ch·kH·kW = 288` columns, filter *f* slicing `[f*9:(f+1)*9]`) the correct value is
> **220/288 = 76.39%**. Layer 1 of the same run reproduces 85.76% exactly, which places the
> fault in that layer's input data rather than in the attack.

## Pipeline

Extraction is driven per filter as a SLURM array; noise is injected **at attack time**, so a
single baseline serves every σ.

```
1. Generate baselines (one job each)
   demo_12m.py    --multipatch   ->  ImageNet activations for both target layers   (~63 GB)
   demo_random.py                 ->  uniform[-2,2] activations, both layers        (~7 GB)
   demo_cifar.py  --multipatch    ->  CIFAR-5M activations for conv1                (~2.5 GB)

2. Layer-1 sweep (array of 32 filters per point; noise via --export=ALL,NOISE=<sigma>)
   extractFilterMaterialized.py                 -> sequential
   extractFilterMaterialized.py --prev_known    -> GT-oracle

3. Layer-2 chain (this is what makes error propagation bite)
   gen_sim_layer2.py   -> hypothesis inputs built from the EXTRACTED layer-1 weights (~59 GB)
   extractFilterMaterialized.py on features.1.0.block.0.0

4. Score and plot
   tools/eval_sweep.py        -> every run dir in one pass
   tools/compare_to_paper.py  -> diff table vs the published values
   tools/make_plot_data.py    -> Figure 19 data, computed from the .pkl files (tools/data/fig19_data.csv)
   tools/make_plots.py        -> Figure 19, drawn from that CSV
```

## Running

```bash
conda activate cpa                          # see the top-level requirements.txt
export IMAGENET_ROOT=/path/to/imagenet      # ImageFolder layout: train/ val/
# CIFAR-5M for the CIFAR experiment: downloaded once into ../data (see the top-level README)
nohup ./run_all.sh > run_all.log 2>&1 &
```

Once the queue drains, score and plot:

```bash
python tools/compare_to_paper.py --repro run
python tools/make_plot_data.py --root run    # -> tools/data/fig19_data.csv
python tools/make_plots.py --outdir run/plots
```

Outputs go to `./run` (override with `RUN=<dir>`). Set `SKIP_CIFAR=1` to omit the CIFAR experiment.

**σ=0 is the correctness gate.** With a correct baseline, noiseless extraction is *exact* —
864/864 bits in both modes. If σ=0 is not 1.0000, the regenerated activations are wrong and
nothing downstream is meaningful; stop and fix that first.

## Tools

| File | Purpose |
|---|---|
| `tools/eval_sweep.py` | Bitwise accuracy for every run directory found under a tree, in one pass. |
| `tools/compare_to_paper.py` | Diff table vs the published values. **Flags any σ point with fewer than 32 filters as `INCOMPLETE`** instead of scoring it. |
| `tools/make_plot_data.py` | Scores Figure 19 **from the `filter_*.pkl` files** into `tools/data/fig19_data.csv`, and refuses to write an incomplete point. |
| `tools/make_plots.py` | Draws Figure 19 and the diff table vs the published values from that CSV (the committed CSV lets you redraw without the raw results). |
| `tools/assert_gpu.py` | Fails a job immediately if it did not get a working GPU. |

Two of these guards exist because of failures observed while producing these results, and are
worth keeping:

- **Partial-result guard.** Every entry point writes its result only at the end, so a lost
  task leaves a *missing* filter rather than a corrupt one. Averaging over the survivors
  yields a plausible-looking number: the CIFAR layer scored **0.3751** on 31/32 filters versus
  **0.3831** on all 32 — close enough to be mistaken for success. `compare_to_paper.py` and
  `make_plot_data.py` therefore refuse to report any point that is not complete.
- **GPU preflight.** Every entry point selects its device with
  `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. On a node whose GPU is
  faulty, `torch` reports no devices and the job runs **silently on CPU, exiting 0** — roughly
  100× slower and with CPU `float16` semantics. `assert_gpu.py` turns that into an immediate,
  loud failure; run it before the payload in every job.

## Notes and caveats

- **Everything is `float16`-centric.** Weights, candidates, leakage and the accuracy metric
  all operate on the 16-bit pattern; the candidate set is every `float16` in `[-5, 5]`.
  Accuracy is the fraction of exactly-matching 16-bit patterns.
- **Noise.** `utils.generate_system_noise_array` adds zero-mean Gaussian noise of standard
  deviation σ to the leakage trace. It also appends **three decoy columns**, noises all four
  and shuffles them; the CPA then takes the **maximum absolute correlation over all four**.
  A candidate can therefore win on a decoy, which is a multiple-comparisons penalty beyond σ
  alone — worth knowing when comparing against a σ-only noise model.
- **Checkpointing.** `checkpoint_every` defaults to 40 weights, but the layers here have 27
  (or 9 per depthwise filter), so no intermediate checkpoint is ever written and a job killed
  mid-filter restarts that filter from scratch. Size wall times with real headroom, or lower
  `--checkpointEvery`.
- **Runtimes** (per filter, one GPU): EfficientNet ≈ 11–20 min; CIFAR `conv1` ≈ 19–40 min.
  Baseline generation: ImageNet ≈ 2 h (I/O-bound — give the job enough memory for the page
  cache), layer-2 hypothesis inputs ≈ 2 h.
- **Peak disk** ≈ 130 GB while the layer-2 baseline and its hypothesis inputs coexist. Both
  are regenerable and can be deleted once results are verified; the results themselves are a
  few hundred MB.
