"""
Post-processing for the paper figures: raw experiment results -> tidy CSVs.

Every notebook in this folder only reads the CSVs written here (./data/<name>.csv)
and plots them; all model loading / GPU work lives in this script.

    python postprocess.py                      # all figures
    python postprocess.py --only zero_noise_graph model_acc_plot
    python postprocess.py --root /path/to/results   # results tree laid out like ../

`--root` is the folder the experiment launchers write into (N0/, N8/, baseLineInps/,
model_cifar.pt, ...); the datasets come from the shared <repo>/data (see ../datasets.py). It defaults to the parent folder, i.e. the layout
produced by run_exps.sh / run_expers_with_noise.sh / run_expers_rnd_models.sh.
"""
import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd

from utilsFuncs import *  # noqa: F401,F403  (loadModel, layerWeightsFunc, torch, nn, ...)
from datasets import CIFAR10_ROOT, CIFAR5M_ROOT  # shared <repo>/data (../datasets.py)

LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'fc1', 'fc2']
LAYER_NUM_FILTERS = [('conv1', 32), ('conv2', 32), ('conv3', 64), ('conv4', 64),
                     ('conv5', 128), ('conv6', 128), ('fc1', 128), ('fc2', 10)]
PGD_SEED = 0
CHKPT_DIR = os.path.join(HERE, 'chkpts')  # preprocess_collision.py checkpoints (large)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def save_csv(rows, out, name, columns):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'{name}.csv')
    rows = [tuple(x.item() if isinstance(x, np.generic) else x for x in r) for r in rows]  # float16 -> exact float
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f'[postprocess] wrote {path} ({len(rows)} rows)')


def zero_noise_model(root, leak_mod, k):
    return os.path.join(root, f'N0/model_extract/cifar6m/{leak_mod}/{k}k/seed8ModelN0{k}k.pth')


def mismatch_prob(model, reference, layer):
    """Fraction of weights in `layer` that differ from the reference model."""
    inCorrect = (layerWeightsFunc(layer, model) != layerWeightsFunc(layer, reference)).sum()
    return inCorrect / layerWeightsFunc(layer, model).size


def cifar10_test_loader(root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_test_data = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True)
    return DataLoader(cifar10_test_data, batch_size=1024, shuffle=True)


def sign_fix(model, loader):
    """Negate fc2 rows of classes with <10% accuracy; keep the flip only if it helps."""
    cloned_model = copy.deepcopy(model)
    class_acc = per_class_accuracy(model, loader, device=device)
    conv_layers = [m for m in cloned_model.modules() if isinstance(m, nn.Linear)]
    last_conv = conv_layers[-1]

    acc_before = calculate_accuracy(model, loader)
    for i, key in enumerate(class_acc.keys()):
        if class_acc[key] < 10:
            with torch.no_grad():
                last_conv.weight.data[i] *= -1
    acc_after = calculate_accuracy(cloned_model, loader)
    if acc_after > acc_before:
        model = cloned_model
    return model


# ---------------------------------------------------------------------------
# one function per notebook
# ---------------------------------------------------------------------------
def zero_noise_graph(root, out):
    baseLineModel = loadModel(os.path.join(root, 'model_cifar.pt'))
    rows = []
    for leak_mod in ['HD', 'HW']:
        for k in [10, 20, 30, 40, 50, 70, 150]:
            model = loadModel(zero_noise_model(root, leak_mod, k))
            for layer in LAYERS:
                rows.append((leak_mod, k, layer, mismatch_prob(model, baseLineModel, layer)))
    save_csv(rows, out, 'zero_noise_graph', ['leakage', 'traces_k', 'layer', 'error_prob'])


def inp_dis_10k_comp(root, out):
    baseLineModel = loadModel(os.path.join(root, 'model_cifar.pt'))
    rows = []
    for leak_mod in ['HD', 'HW']:
        for inpu in ['random', 'c6m_low_acc', 'cifar10', 'cifar6m', 'c6m_high_acc']:
            model = loadModel(os.path.join(root, f'N0/inp_dis_comp/{inpu}/{leak_mod}/10k/seed8ModelN010k.pth'))
            for layer in LAYERS:
                rows.append((leak_mod, inpu, layer, mismatch_prob(model, baseLineModel, layer)))
    save_csv(rows, out, 'inp_dis_10k_comp', ['leakage', 'input_dist', 'layer', 'error_prob'])


def model_weight_dist_comp(root, out):
    # (extracted model, the baseline it was extracted from)
    pairs = [('randUniform', 'N0/rnd_model/randUniform/HD/10k/seed8ModelN010k.pth', 'model_rand_uniform.pth'),
             ('randInit', 'N0/rnd_model/randInit/HD/10k/seed8ModelN010k.pth', 'model_rand_init.pth'),
             ('pretrained', 'N0/inp_dis_comp/random/HD/10k/seed8ModelN010k.pth', 'model_cifar.pt')]
    rows = []
    for name, extracted, baseline in pairs:
        randModel = loadModel(os.path.join(root, extracted))
        baselineModel = loadModel(os.path.join(root, baseline))
        for layer in LAYERS:
            rows.append((name, layer, mismatch_prob(randModel, baselineModel, layer)))
    save_csv(rows, out, 'model_weight_dist_comp', ['baseline_model', 'layer', 'error_prob'])

    # fraction of exact-zero inputs per weight position of conv4 (rand-init model)
    M = np.asarray(pd.read_pickle(os.path.join(root, 'baseLineInps/rnd_model/HD/randInit/10000/conv4.pkl')))
    finite = np.isfinite(M)
    denom = finite.sum(axis=0)
    num0 = ((M == 0) & finite).sum(axis=0)
    p0 = np.full(M.shape[1], np.nan)
    mask = denom > 0
    p0[mask] = num0[mask] / denom[mask]
    save_csv(list(enumerate(p0)), out, 'model_weight_dist_comp_zero_inputs', ['weight_index', 'zero_input_frac'])


def model_acc_plot(root, out):
    loader = cifar10_test_loader(root)
    rows = []
    for leak_mod in ['HD', 'HW']:
        for k in [2, 3, 4, 5, 6, 7, 10, 20]:
            model = loadModel(zero_noise_model(root, leak_mod, k))
            acc_raw = calculate_accuracy(model, loader) / 100
            acc_fixed = calculate_accuracy(sign_fix(model, loader), loader) / 100
            rows.append((leak_mod, k, acc_raw, acc_fixed))
    save_csv(rows, out, 'model_acc_plot', ['leakage', 'traces_k', 'acc_raw', 'acc_sign_fixed'])


NOISE_SPECS = [  # (traces_k, leakage, noise levels)
    (200, 'HD', [4, 6, 10, 12, 14, 16]),
    (200, 'HW', [4, 6, 10, 12, 14, 16]),
    (400, 'HD', [4, 6, 10, 12, 14, 16]),
    (400, 'HW', [4, 6, 10, 12, 14, 16]),
    (800, 'HD', [4, 6, 10, 12, 14, 16]),
    (800, 'HW', [8, 10, 12, 14]),
]


def noise_ext_comp(root, out):
    loader = cifar10_test_loader(root)
    rows = []
    for k, leak_mod, noises in NOISE_SPECS:
        for n in noises:
            model = loadModel(os.path.join(root, f'N{n}/model_extract/cifar6m/{leak_mod}/{k}k/seed8ModelN{n}{k}k.pth'))
            acc_raw = calculate_accuracy(model, loader) / 100.0
            acc_fixed = calculate_accuracy(sign_fix(model, loader), loader) / 100.0
            rows.append((leak_mod, k, n, acc_raw, acc_fixed))
    save_csv(rows, out, 'noise_ext_comp', ['leakage', 'traces_k', 'sigma', 'acc_raw', 'acc_sign_fixed'])


# --- pgd_attack_exp ---------------------------------------------------------
# PGD transfer at the eps where the white-box attack is ~100% successful:
# 1) white-box eps sweep on the baseline -> EPS = smallest eps in PGD_EPS_GRID with ASR >= 0.99;
# 2) at EPS, PGD crafted on each extracted model (after the fc sign-flip fix), on the
#    non-extracted surrogate and on the baseline itself, transferred to the baseline.
# alpha = 2/255, 75 steps, random start, untargeted. The attack runs on fp32 copies of the
# models with fp64 cross-entropy: in fp16 PGD stalls on confident models (the gradient
# underflows), which caps the white-box ASR below 100% at any eps.
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
PGD_EPS_GRID = [2, 4, 5, 6, 8, 10, 12, 16]          # in units of 1/255
PGD_NUM_TRACES = [2, 3, 4, 5, 6, 7, 10, 20]


def denormalize(x):
    return x * STD.to(x.device) + MEAN.to(x.device)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def pgd(model, x_norm, y, eps, alpha, steps):
    x0 = denormalize(x_norm).detach()
    x_adv = torch.clamp(x0 + torch.empty_like(x0).uniform_(-eps, eps), 0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(normalize(x_adv)).double(), y)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.clamp(torch.max(torch.min(x_adv, x0 + eps), x0 - eps), 0, 1).detach()
    return normalize(x_adv)


def adv_serial_transfer(source, target, loader, eps, alpha=2 / 255, steps=75):
    source = copy.deepcopy(source).float().eval()
    target = copy.deepcopy(target).float().eval()
    tot = src_clean = src_adv = tgt_adv = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device).float(), labels.to(device)
        with torch.no_grad():
            src_clean += (source(imgs).argmax(1) == labels).sum().item()
        adv = pgd(source, imgs, labels, eps, alpha, steps)
        with torch.no_grad():
            src_adv += (source(adv).argmax(1) == labels).sum().item()
            tgt_adv += (target(adv).argmax(1) == labels).sum().item()
        tot += labels.numel()
    return {'source_clean_acc': src_clean / tot, 'source_adv_acc': src_adv / tot,
            'target_adv_acc': tgt_adv / tot, 'asr': 1 - tgt_adv / tot}


def load_fixed(path, loader):
    """sign_fix that also returns the accuracy. Kept separate (same loader passes as the
    paper run) because every DataLoader pass draws from the torch RNG that seeds the PGD
    random starts."""
    model = loadModel(path)
    cloned = copy.deepcopy(model)
    class_acc = per_class_accuracy(model, loader, device=device)
    last = [m for m in cloned.modules() if isinstance(m, nn.Linear)][-1]
    acc_before = calculate_accuracy(model, loader)
    for i, key in enumerate(class_acc.keys()):
        if class_acc[key] < 10:
            with torch.no_grad():
                last.weight.data[i] *= -1
    acc_after = calculate_accuracy(cloned, loader)
    return (cloned if acc_after > acc_before else model), max(acc_before, acc_after)


def pgd_attack_exp(root, out, target_white=0.99):
    torch.manual_seed(PGD_SEED)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    loader = DataLoader(CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform, download=True),
                        batch_size=1024, shuffle=False)
    base = loadModel(os.path.join(root, 'model_cifar.pt'))
    surrogate = loadModel(os.path.join(root, 'non_extracted_surrogate_cifar_model.pt'))

    rows = []
    sweep = {}
    for e in PGD_EPS_GRID:
        sweep[e] = adv_serial_transfer(base, base, loader, e / 255)['asr']
        rows.append(('white_box_sweep', None, e, sweep[e], None))
        print(f'white box eps {e}/255: ASR {sweep[e]:.4f}', flush=True)
        if sweep[e] >= target_white:
            break
    eps = next((e for e in PGD_EPS_GRID if sweep.get(e, 0) >= target_white), PGD_EPS_GRID[-1])
    print(f'-> eps = {eps}/255', flush=True)

    rows.append(('white_box', None, eps, adv_serial_transfer(base, base, loader, eps / 255)['asr'], None))
    rows.append(('surrogate', None, eps, adv_serial_transfer(surrogate, base, loader, eps / 255)['asr'], None))
    for leak_mod in ('HD', 'HW'):
        for k in PGD_NUM_TRACES:
            model, acc = load_fixed(zero_noise_model(root, leak_mod, k), loader)
            asr = adv_serial_transfer(model, base, loader, eps / 255)['asr']
            rows.append((leak_mod, k, eps, asr, acc / 100))
            print(f'{leak_mod} {k}k: acc {acc:.4f}  ASR on target {asr:.4f}', flush=True)
    save_csv(rows, out, 'pgd_attack_exp', ['source', 'traces_k', 'eps_255', 'asr', 'clean_acc'])


# --- inp_errs_probs ---------------------------------------------------------
def load_cifar5m_part(root, part='part0.npz'):
    """Same preprocessing as the extraction pipeline: uint8 -> PIL -> ToTensor -> Normalize(0.5)."""
    from torch.utils.data import Dataset

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    class CIFAR5MDataset(Dataset):
        def __init__(self, path):
            data = np.load(path)
            self.images, self.labels = data["X"], data["Y"]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
            img = transforms.ToPILImage()(img)
            return transform(img), int(self.labels[idx])

    loader = DataLoader(CIFAR5MDataset(os.path.join(CIFAR5M_ROOT, part)),
                        batch_size=2_000_000, shuffle=False, num_workers=40)
    images, _ = next(iter(loader))
    return images


def get_inputs_outputs(model, inputTensor, layer_name):
    hooks = setupHooks(model)
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)) and name == layer_name:
            modelAInputs, modelAoutputs = calc_layer_Input_output_in_batches(model, inputTensor, layer)
    removeHooks(hooks)
    return modelAInputs, modelAoutputs


def compareModels(modelA, modelB, inputs):
    """Fraction of differing inputs/outputs of every Conv2d/Linear/BatchNorm2d layer."""
    lstTBoth = []
    for name, layer in modelA.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            modelAInputs, modelAoutputs = get_inputs_outputs(modelA, inputs, name)
            modelBInputs, modelBoutputs = get_inputs_outputs(modelB, inputs, name)
            lstTBoth.extend([np.count_nonzero(modelAInputs != modelBInputs) / modelAInputs.size,
                             np.count_nonzero(modelAoutputs != modelBoutputs) / modelAoutputs.size])
    return lstTBoth


def inp_errs_probs(root, out):
    # fp16 activations through cuDNN: the exact mismatch fractions depend on the GPU
    # architecture (e.g. Pascal vs Ampere differ by up to ~0.04), not only on the models
    baseLineModel = loadModel(os.path.join(root, 'model_cifar.pt'))
    input_tensor = load_cifar5m_part(root)
    rows = []
    for leak_mod in ['HD', 'HW']:
        for k in [2, 4, 6, 10, 30]:
            print(f'Processing {leak_mod} model trace: {k}')
            model = loadModel(zero_noise_model(root, leak_mod, k))
            inputs = input_tensor[: int(k * 1000), ].half().to(device)
            combinedDiff = compareModels(baseLineModel, model, inputs)
            # drop entries that duplicate a neighbour (e.g. BN input == ReLU output)
            filtered = [x for i, x in enumerate(combinedDiff) if i not in [0, 3, 3 + 8, 3 + 8 + 8, 27]]
            rows += [(leak_mod, k, i, x) for i, x in enumerate(filtered)]
    save_csv(rows, out, 'inp_errs_probs', ['leakage', 'traces_k', 'stage_idx', 'err_prob'])


# --- ambguity_exp -----------------------------------------------------------
def ambguity_exp(root, out):
    baseLineModel = loadModel(os.path.join(root, 'model_cifar.pt'))
    baselineInps = pd.read_pickle(os.path.join(root, 'baseLineInps/model_extract/HD/cifar6m/10000/conv4.pkl'))
    rows = []
    for index in [0, 5, 100, 200, 450]:
        correctWeights = layerWeightsFunc('conv4', baseLineModel)
        baselinePrevsums = np.array([compute_prevSum_HW(baselineInps, correctWeights[i, :], leakage_model='HD')[0] for i in range(64)])
        for eps_sign, direction in [('plus', np.inf), ('minus', -np.inf)]:
            conv4Weights = layerWeightsFunc('conv4', baseLineModel)
            conv4Weights[:, index] = np.nextafter(conv4Weights[:, index], np.float16(direction), dtype=np.float16)
            baseline_prev_sum_mod = np.array([compute_prevSum_HW(baselineInps, conv4Weights[i, :], leakage_model='HD')[0] for i in range(64)])
            rate = (baseline_prev_sum_mod[:, :, index:100 + index] == baselinePrevsums[:, :, index:100 + index]).mean(axis=1).mean(axis=0)
            rows += [(eps_sign, index, offset, r) for offset, r in enumerate(rate)]
    save_csv(rows, out, 'ambguity_exp', ['eps_sign', 'weight_index', 'offset', 'collision_rate'])


# --- err_collision ----------------------------------------------------------
def mean_with_masked_arrays(arrays):
    max_len = max(len(arr) for arr in arrays)
    mask_arr = np.ma.masked_all((len(arrays), max_len), dtype=float)
    for i, arr in enumerate(arrays):
        mask_arr[i, :len(arr)] = arr
    return np.ma.mean(mask_arr, axis=0).filled(np.nan)


def fp16_exp(x_fp16):
    """FP16 exponent extractor (returns 0..31)."""
    u16 = np.asarray(x_fp16, dtype=np.float16).view(np.uint16)
    return ((u16 >> 10) & 0x1F).astype(np.int16)


def compute_delta_exp_histograms(eps_prev_sum_equality_rate, eps_multi_equality_rate, prevSums,
                                 baseline_multi_candidate, delta_range=(-31, 31)):
    """
    Histograms over de = exp(prev_sum[..., :-1]) - exp(mul[..., 1:]):
      - hist_equiv:   prev-sum equality True and multiplication equality False (collision)
      - hist_unequiv: prev-sum equality False (no collision)
    """
    MIN_D, MAX_D = delta_range
    NBINS = MAX_D - MIN_D + 1
    hist_equiv = np.zeros(NBINS, dtype=np.int64)
    hist_unequiv = np.zeros(NBINS, dtype=np.int64)

    for i, (layer, _) in enumerate(LAYER_NUM_FILTERS):
        print(f"Processing layer {layer}...")
        sum_eq_layer = np.asarray(eps_prev_sum_equality_rate[i], dtype=bool)
        current_sum = np.asarray(prevSums[i], dtype=np.float16)
        multi = np.asarray(baseline_multi_candidate[i], dtype=np.float16)
        mul_eq_layer = np.asarray(eps_multi_equality_rate[i], dtype=bool)

        # Align: prev_sum[..., :-1] corresponds to multi[..., 1:]
        de = (fp16_exp(current_sum[..., :-1]) - fp16_exp(multi[..., 1:])).astype(np.int16)
        sum_eq_part = sum_eq_layer[..., 1:]
        mul_eq_part = mul_eq_layer[..., 1:]

        in_range = (de >= MIN_D) & (de <= MAX_D)
        idx = (de[in_range] - MIN_D).astype(np.int32)
        sum_eq_part = sum_eq_part[in_range]
        mul_eq_part = mul_eq_part[in_range]

        idx_eq = idx[sum_eq_part & (~mul_eq_part)]
        if idx_eq.size:
            hist_equiv += np.bincount(idx_eq, minlength=NBINS)
        idx_uneq = idx[~sum_eq_part]
        if idx_uneq.size:
            hist_unequiv += np.bincount(idx_uneq, minlength=NBINS)

    s_eq, s_uneq = hist_equiv.sum(), hist_unequiv.sum()
    prob_equiv = hist_equiv / s_eq if s_eq else hist_equiv.astype(np.float64)
    prob_unequiv = hist_unequiv / s_uneq if s_uneq else hist_unequiv.astype(np.float64)
    xs = np.arange(MIN_D, MAX_D + 1, dtype=np.int16)
    return xs, hist_equiv, hist_unequiv, prob_equiv, prob_unequiv


def err_collision(root, out, prev_sum_bins=40):
    from preprocess_collision import CollisionConfig, ensure_required_pickles, load_pkl

    baseline_inputs_root = os.path.join(root, 'baseLineInps/model_extract/HD/cifar6m/10000')
    cfg = CollisionConfig(chkpt_dir=CHKPT_DIR, model_path=os.path.join(root, 'model_cifar.pt'),
                          baseline_inputs_root=baseline_inputs_root)
    ensure_required_pickles(cfg, force=False)
    baseLineModel = loadModel(cfg.model_path)

    # (a) collision rate per weight index, averaged over layers
    series = [('Partial Sum HW', 'means_hw.pkl'), ('Partial Sum HD', 'means_hd.pkl'),
              ('Partial Sum Value', 'means_intermediate_sum.pkl'), ('Multipication Value', 'multi_means.pkl')]
    rows = []
    for label, pkl in series:
        rows += [(label, i, p) for i, p in enumerate(mean_with_masked_arrays(load_pkl(cfg, pkl)))]
    save_csv(rows, out, 'err_collision_rate', ['series', 'weight_index', 'collision_rate'])

    # (b) multiplication collisions by float16 mantissa (max of +eps / -eps per weight)
    plus_eps_multi_equality_rate = load_pkl(cfg, "eq_mult_value_plus.pkl")
    minus_eps_multi_equality_rate = load_pkl(cfg, "eq_mult_value_minus.pkl")
    coll_by_mant = np.zeros(1024, dtype=np.float64)
    total_by_mant = np.zeros(1024, dtype=np.float64)
    for i, (layer, num_filters) in enumerate(LAYER_NUM_FILTERS):
        print(f"Processing layer {layer}...")
        inputs_baseline = pd.read_pickle(os.path.join(baseline_inputs_root, f'{layer}.pkl'))
        correct_candidates = layerWeightsFunc(layer, baseLineModel).astype(np.float16, copy=False)
        plus_mask = plus_eps_multi_equality_rate[i]    # (num_filters, num_inputs, weight_dim)
        minus_mask = minus_eps_multi_equality_rate[i]
        assert plus_mask.shape == minus_mask.shape
        assert plus_mask.shape[0] == num_filters
        assert plus_mask.shape[1] == inputs_baseline.shape[0]

        best_counts_xz = np.maximum(plus_mask.sum(axis=1).astype(np.float64),
                                    minus_mask.sum(axis=1).astype(np.float64))
        total_counts_xz = np.full(best_counts_xz.shape, float(plus_mask.shape[1]), dtype=np.float64)
        mant_flat = (correct_candidates.view(np.uint16) & 0x03FF).astype(np.int32).ravel()
        coll_by_mant += np.bincount(mant_flat, weights=best_counts_xz.ravel(), minlength=1024)
        total_by_mant += np.bincount(mant_flat, weights=total_counts_xz.ravel(), minlength=1024)
    rows = [(m, coll_by_mant[m], total_by_mant[m]) for m in range(1024)]
    save_csv(rows, out, 'err_collision_mantissa', ['mantissa', 'collisions', 'total'])

    # (c) exponent difference prev-sum vs. product, collision vs. no-collision
    prevSums = load_pkl(cfg, "prev_sums_ckpt.pkl")
    baseline_multi_candidate = load_pkl(cfg, "baseline_multi_candidate.pkl")
    rows = []
    for eps_sign, mult_eq in [('plus', plus_eps_multi_equality_rate), ('minus', minus_eps_multi_equality_rate)]:
        prev_eq = load_pkl(cfg, f"eq_prevsum_value_{eps_sign}.pkl")
        xs, hist_eq, hist_uneq, prob_eq, prob_uneq = compute_delta_exp_histograms(
            prev_eq, mult_eq, prevSums, baseline_multi_candidate, delta_range=(-31, 31))
        del prev_eq
        rows += list(zip([eps_sign] * len(xs), xs, hist_eq, hist_uneq, prob_eq, prob_uneq))
    del prevSums, baseline_multi_candidate, plus_eps_multi_equality_rate, minus_eps_multi_equality_rate
    save_csv(rows, out, 'err_collision_delta_exp', ['eps_sign', 'exp_diff', 'count_collision', 'count_no_collision',
                                                    'prob_collision', 'prob_no_collision'])

    # (d) |partial sum| growth along fc1 (mean over inputs, per filter); only the
    #     log-spaced weight indices that the boxplot shows are stored
    inputs_baseline = pd.read_pickle(os.path.join(baseline_inputs_root, 'fc1.pkl'))
    fc1 = layerWeightsFunc('fc1', baseLineModel)
    prev_sums = np.array([compute_prevSum_HW(inputs_baseline, fc1[i, :], leakage_model='HW')[0] for i in range(128)])
    res = np.abs(prev_sums).mean(axis=1)                       # (filters, weight_dim)
    W = res.shape[1]
    idxs = np.insert(np.unique(np.rint(np.geomspace(1, W - 1, prev_sum_bins)).astype(int)), 0, 0)
    rows = [(f, int(w), res[f, w]) for f in range(res.shape[0]) for w in idxs]
    save_csv(rows, out, 'err_collision_prev_sum', ['filter', 'weight_index', 'mean_abs_prev_sum'])


# --- weight0HWDist ----------------------------------------------------------
def f16_exp_shift_candidates(w, deltas=(-8, -4, -2, -1, 1, 2, 4, 8), min_val=-5.0, max_val=5.0):
    """{delta: float16(w * 2**delta)} for every candidate inside [min_val, max_val]."""
    w16 = np.asarray(w, dtype=np.float16)
    d = np.asarray(deltas, dtype=np.int16)
    vals32 = np.ldexp(w16.astype(np.float32)[..., None], d)
    vals16 = vals32.astype(np.float16)
    mask = (vals32 >= min_val) & (vals32 <= max_val) & np.isfinite(vals32)
    return {int(di): np.float16(vi) for di, vi, m in zip(d, vals16.ravel(), mask.ravel()) if m}


def weight0HWDist(root, out):
    baseLineModel = loadModel(os.path.join(root, 'model_cifar.pt'))
    rows = []
    for layer in ['conv1', 'conv4', 'conv6', 'fc1']:
        layer_diffs = []
        inputs_baseline = pd.read_pickle(os.path.join(root, f'baseLineInps/model_extract/HD/cifar6m/10000/{layer}.pkl'))[:, 0]
        for correct_ind0 in layerWeightsFunc(layer, baseLineModel)[:, 0]:
            candidates = f16_exp_shift_candidates(correct_ind0)
            cand_corrs = {cand: comp_corr(compute_hyp(cand, inputs_baseline), compute_hyp(correct_ind0, inputs_baseline))
                          for delta, cand in candidates.items()}
            max_cand = max(cand_corrs, key=cand_corrs.get)
            a = np.asarray(compute_hyp(max_cand, inputs_baseline) - compute_hyp(correct_ind0, inputs_baseline))
            layer_diffs.extend(-a if np.count_nonzero(a < 0) > np.count_nonzero(a > 0) else a)
        values, counts = np.unique(np.asarray(layer_diffs), return_counts=True)
        rows += [(layer, v, c) for v, c in zip(values, counts)]
    save_csv(rows, out, 'weight0HWDist', ['layer', 'hw_diff', 'count'])


# --- zero_padding -----------------------------------------------------------
def zero_padding(root, out):
    inputs = pd.read_pickle(os.path.join(root, 'baseLineInps/model_extract/HD/cifar6m/6000/fc2.pkl'))
    w = 1.11511230e-01
    rows = []
    for pad in [0, 1, 10, 50, 100, 150, 200, 1000, 10000, 100000, 1000000, 10_000_000, 100_000_000]:
        new_inputs = np.pad(inputs[inputs != 0], (0, pad), mode='constant', constant_values=0)
        rows.append((pad, comp_corr(compute_hyp(-w, new_inputs), compute_hyp(w, new_inputs))))
    save_csv(rows, out, 'zero_padding', ['zero_padding', 'correlation'])


FIGURES = {f.__name__: f for f in [zero_noise_graph, inp_dis_10k_comp, model_weight_dist_comp, model_acc_plot,
                                   noise_ext_comp, pgd_attack_exp, inp_errs_probs, ambguity_exp,
                                   err_collision, weight0HWDist, zero_padding]}


def main():
    global CHKPT_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=os.path.dirname(HERE), help='experiment results folder (default: ..)')
    parser.add_argument('--out', default=os.path.join(HERE, 'data'), help='CSV output folder (default: ./data)')
    parser.add_argument('--chkpts', default=CHKPT_DIR, help='preprocess_collision.py checkpoint folder (default: ./chkpts)')
    parser.add_argument('--only', nargs='+', choices=list(FIGURES), help='run only these notebooks\' post-processing')
    args = parser.parse_args()
    root, out = os.path.abspath(args.root), os.path.abspath(args.out)
    CHKPT_DIR = os.path.abspath(args.chkpts)
    for name in args.only or FIGURES:
        print(f'[postprocess] {name}')
        FIGURES[name](root, out)


if __name__ == '__main__':
    main()
