"""
Deletion/Insertion AUC Test for Attribution Reliability Verification.

Reference: Zhang et al. (INTERSPEECH 2023) used mask+EER to verify attribution reliability.
We extend this with Deletion/Insertion AUC, a more fine-grained and efficient approach
that doesn't require retraining.

Key idea:
- Deletion: progressively remove high-attribution features → if attribution is correct,
  removing top features should cause rapid performance degradation.
- Insertion: progressively add high-attribution features → if attribution is correct,
  adding top features should quickly recover performance.
- Random baseline: same deletion/insertion but with random ordering → much weaker effect.

Metrics:
- Deletion AUC: area under the EER-vs-deletion-ratio curve. Higher = better attribution.
- Insertion AUC: area under the EER-vs-insertion-ratio curve. Higher = better attribution.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .analyzer import load_audio_as_tensor
from .integrated_gradients import IntegratedGradients_ECAPA


# ──────────────────────────────────────────────
#  Core: deletion / insertion on FBank
# ──────────────────────────────────────────────

def _apply_deletion(fbank_tensor, attribution, ratio, mode='freq_time'):
    """
    Zero out the top-`ratio` fraction of attribution locations in fbank_tensor.

    Args:
        fbank_tensor: [1, 80, T] FBank features (on device, with grad disabled)
        attribution:  [80, T] or [1, 80, T] attribution map (numpy)
        ratio: fraction of features to delete [0, 1]
        mode: 'freq_time' — delete individual (freq, time) cells
              'freq'      — delete entire frequency bins (all time steps)
              'time'      — delete entire time frames (all freq bins)

    Returns:
        modified_fbank: [1, 80, T] tensor with top-ratio positions zeroed
    """
    if attribution.ndim == 3 and attribution.shape[0] == 1:
        attribution = attribution.squeeze(0)
    modified = fbank_tensor.clone()
    attr_flat = attribution.flatten()

    if mode == 'freq_time':
        threshold = np.percentile(np.abs(attr_flat), (1 - ratio) * 100)
        mask = np.abs(attribution) >= threshold
        modified[0, mask] = 0.0

    elif mode == 'freq':
        # attribution per frequency bin: sum over time
        freq_importance = np.abs(attribution).sum(axis=1)  # [80]
        n_delete = max(1, int(ratio * len(freq_importance)))
        top_bins = np.argsort(freq_importance)[::-1][:n_delete]
        modified[0, top_bins, :] = 0.0

    elif mode == 'time':
        # attribution per time frame: sum over freq
        time_importance = np.abs(attribution).sum(axis=0)  # [T]
        n_delete = max(1, int(ratio * len(time_importance)))
        top_frames = np.argsort(time_importance)[::-1][:n_delete]
        modified[0, :, top_frames] = 0.0

    return modified


def _apply_insertion(fbank_tensor, attribution, ratio, mode='freq_time'):
    """
    Keep only the top-`ratio` fraction of attribution locations in fbank_tensor,
    zero everything else.

    Args: same as _apply_deletion
    Returns:
        modified_fbank: [1, 80, T] tensor with only top-ratio positions kept
    """
    if attribution.ndim == 3 and attribution.shape[0] == 1:
        attribution = attribution.squeeze(0)
    modified = torch.zeros_like(fbank_tensor)
    attr_flat = attribution.flatten()

    if mode == 'freq_time':
        threshold = np.percentile(np.abs(attr_flat), (1 - ratio) * 100)
        mask = np.abs(attribution) >= threshold
        modified[0, mask] = fbank_tensor[0, mask]

    elif mode == 'freq':
        freq_importance = np.abs(attribution).sum(axis=1)
        n_keep = max(1, int(ratio * len(freq_importance)))
        top_bins = np.argsort(freq_importance)[::-1][:n_keep]
        modified[0, top_bins, :] = fbank_tensor[0, top_bins, :]

    elif mode == 'time':
        time_importance = np.abs(attribution).sum(axis=0)
        n_keep = max(1, int(ratio * len(time_importance)))
        top_frames = np.argsort(time_importance)[::-1][:n_keep]
        modified[0, :, top_frames] = fbank_tensor[0, :, top_frames]

    return modified


def _apply_random_deletion(fbank_tensor, ratio, rng=None, mode='freq_time'):
    """Same as _apply_deletion but with random ordering (baseline)."""
    if rng is None:
        rng = np.random.default_rng()
    modified = fbank_tensor.clone()
    F, T = fbank_tensor.shape[1], fbank_tensor.shape[2]

    if mode == 'freq_time':
        n_total = F * T
        n_delete = int(ratio * n_total)
        flat = modified[0].flatten()
        indices = rng.choice(n_total, n_delete, replace=False)
        flat[indices] = 0.0
        modified[0] = flat.view(F, T)

    elif mode == 'freq':
        n_delete = max(1, int(ratio * F))
        bins = rng.choice(F, n_delete, replace=False)
        modified[0, bins, :] = 0.0

    elif mode == 'time':
        n_delete = max(1, int(ratio * T))
        frames = rng.choice(T, n_delete, replace=False)
        modified[0, :, frames] = 0.0

    return modified


def _apply_random_insertion(fbank_tensor, ratio, rng=None, mode='freq_time'):
    """Same as _apply_insertion but with random ordering (baseline)."""
    if rng is None:
        rng = np.random.default_rng()
    modified = torch.zeros_like(fbank_tensor)
    F, T = fbank_tensor.shape[1], fbank_tensor.shape[2]

    if mode == 'freq_time':
        n_total = F * T
        n_keep = max(1, int(ratio * n_total))
        flat_orig = fbank_tensor[0].flatten()
        flat_mod = modified[0].flatten()
        indices = rng.choice(n_total, n_keep, replace=False)
        flat_mod[indices] = flat_orig[indices]
        modified[0] = flat_mod.view(F, T)

    elif mode == 'freq':
        n_keep = max(1, int(ratio * F))
        bins = rng.choice(F, n_keep, replace=False)
        modified[0, bins, :] = fbank_tensor[0, bins, :]

    elif mode == 'time':
        n_keep = max(1, int(ratio * T))
        frames = rng.choice(T, n_keep, replace=False)
        modified[0, :, frames] = fbank_tensor[0, :, frames]

    return modified


# ──────────────────────────────────────────────
#  Forward from FBank (same as IG)
# ──────────────────────────────────────────────

def _forward_from_fbank(model, fbank_tensor):
    """Forward ECAPA-TDNN from FBank tensor [1, 80, T] → embedding [1, 192]."""
    if fbank_tensor.dim() == 4 and fbank_tensor.shape[1] == 1:
        fbank_tensor = fbank_tensor.squeeze(1)

    x = model.conv1(fbank_tensor)
    x = model.relu(x)
    x = model.bn1(x)

    x1 = model.layer1(x)
    x2 = model.layer2(x + x1)
    x3 = model.layer3(x + x1 + x2)

    x = model.layer4(torch.cat((x1, x2, x3), dim=1))
    x = model.relu(x)

    t = x.size()[-1]
    global_x = torch.cat((
        x,
        torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
        torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
    ), dim=1)

    w = model.attention(global_x)
    mu = torch.sum(x * w, dim=2)
    sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
    x = torch.cat((mu, sg), 1)
    x = model.bn5(x)
    x = model.fc6(x)
    x = model.bn6(x)
    return x


# ──────────────────────────────────────────────
#  Cosine similarity scoring
# ──────────────────────────────────────────────

def compute_score(model, fbank_target, fbank_ref):
    """Compute cosine similarity score between target and reference."""
    with torch.no_grad():
        emb_target = _forward_from_fbank(model, fbank_target)
        emb_ref = _forward_from_fbank(model, fbank_ref)
        emb_target = F.normalize(emb_target, p=2, dim=1)
        emb_ref = F.normalize(emb_ref, p=2, dim=1)
        score = torch.sum(emb_target * emb_ref, dim=1).item()
    return score


# ──────────────────────────────────────────────
#  Main: Deletion/Insertion Test
# ──────────────────────────────────────────────

def deletion_insertion_test(
    model,
    target_tensor: torch.Tensor,
    ref_same_tensor: torch.Tensor,
    ref_diff_tensor: torch.Tensor,
    attribution: np.ndarray,
    ratios: List[float] = None,
    mode: str = 'freq_time',
    n_random: int = 10,
    device: str = 'cuda',
) -> Dict:
    """
    Run Deletion/Insertion test for a single model + single sample.

    Args:
        model: ECAPA_TDNN speaker encoder
        target_tensor: [1, samples] target audio
        ref_same_tensor: [1, samples] same-speaker reference
        ref_diff_tensor: [1, samples] different-speaker reference
        attribution: [80, T] attribution map (e.g., cosine_sim_diff)
        ratios: list of deletion/insertion ratios
        mode: 'freq_time', 'freq', or 'time'
        n_random: number of random baseline runs to average
        device: compute device

    Returns:
        dict with keys: 'deletion', 'insertion', 'deletion_random', 'insertion_random',
                        'original_score_same', 'original_score_diff', 'ratios'
    """
    if ratios is None:
        ratios = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    model.eval()

    # Extract FBank features
    with torch.no_grad():
        fbank_target = model.torchfbank(target_tensor) + 1e-6
        fbank_target = fbank_target.log()
        fbank_target = fbank_target - torch.mean(fbank_target, dim=-1, keepdim=True)

        fbank_ref_same = model.torchfbank(ref_same_tensor) + 1e-6
        fbank_ref_same = fbank_ref_same.log()
        fbank_ref_same = fbank_ref_same - torch.mean(fbank_ref_same, dim=-1, keepdim=True)

        fbank_ref_diff = model.torchfbank(ref_diff_tensor) + 1e-6
        fbank_ref_diff = fbank_ref_diff.log()
        fbank_ref_diff = fbank_ref_diff - torch.mean(fbank_ref_diff, dim=-1, keepdim=True)

    # Original scores
    original_score_same = compute_score(model, fbank_target, fbank_ref_same)
    original_score_diff = compute_score(model, fbank_target, fbank_ref_diff)

    # ── Deletion ──
    deletion_scores_same = {}
    deletion_scores_diff = {}

    for ratio in ratios:
        modified = _apply_deletion(fbank_target, attribution, ratio, mode=mode)
        score_same = compute_score(model, modified, fbank_ref_same)
        score_diff = compute_score(model, modified, fbank_ref_diff)
        deletion_scores_same[ratio] = score_same
        deletion_scores_diff[ratio] = score_diff

    # ── Insertion ──
    insertion_scores_same = {}
    insertion_scores_diff = {}

    for ratio in ratios:
        modified = _apply_insertion(fbank_target, attribution, ratio, mode=mode)
        score_same = compute_score(model, modified, fbank_ref_same)
        score_diff = compute_score(model, modified, fbank_ref_diff)
        insertion_scores_same[ratio] = score_same
        insertion_scores_diff[ratio] = score_diff

    # ── Random baselines ──
    rng = np.random.default_rng(42)
    random_deletion_same = {r: [] for r in ratios}
    random_deletion_diff = {r: [] for r in ratios}
    random_insertion_same = {r: [] for r in ratios}
    random_insertion_diff = {r: [] for r in ratios}

    for _ in range(n_random):
        for ratio in ratios:
            mod_del = _apply_random_deletion(fbank_target, ratio, rng=rng, mode=mode)
            random_deletion_same[ratio].append(compute_score(model, mod_del, fbank_ref_same))
            random_deletion_diff[ratio].append(compute_score(model, mod_del, fbank_ref_diff))

            mod_ins = _apply_random_insertion(fbank_target, ratio, rng=rng, mode=mode)
            random_insertion_same[ratio].append(compute_score(model, mod_ins, fbank_ref_same))
            random_insertion_diff[ratio].append(compute_score(model, mod_ins, fbank_ref_diff))

    # Average random baselines
    avg_random_deletion_same = {r: np.mean(v) for r, v in random_deletion_same.items()}
    avg_random_deletion_diff = {r: np.mean(v) for r, v in random_deletion_diff.items()}
    avg_random_insertion_same = {r: np.mean(v) for r, v in random_insertion_same.items()}
    avg_random_insertion_diff = {r: np.mean(v) for r, v in random_insertion_diff.items()}

    return {
        'ratios': ratios,
        'original_score_same': original_score_same,
        'original_score_diff': original_score_diff,
        'deletion_scores_same': deletion_scores_same,
        'deletion_scores_diff': deletion_scores_diff,
        'insertion_scores_same': insertion_scores_same,
        'insertion_scores_diff': insertion_scores_diff,
        'random_deletion_same': avg_random_deletion_same,
        'random_deletion_diff': avg_random_deletion_diff,
        'random_insertion_same': avg_random_insertion_same,
        'random_insertion_diff': avg_random_insertion_diff,
    }


# ──────────────────────────────────────────────
#  Batch: Run over many pairs, compute EER at each ratio
# ──────────────────────────────────────────────

def batch_deletion_insertion_test(
    model,
    sample_pairs: List[Dict],
    attribution_method: str = 'cosine_sim_diff',
    ratios: List[float] = None,
    mode: str = 'freq_time',
    n_random: int = 10,
    n_steps: int = 50,
    device: str = 'cuda',
    eval_path: str = None,
) -> Dict:
    """
    Run Deletion/Insertion test over many sample pairs and compute EER at each ratio.

    Args:
        model: ECAPA_TDNN speaker encoder
        sample_pairs: list of dicts with keys 'target', 'ref_same', 'ref_diff', 'label'
        attribution_method: 'cosine_sim_diff' or 'l2_norm'
        ratios: deletion/insertion ratios
        mode: 'freq_time', 'freq', or 'time'
        n_random: number of random baseline runs
        n_steps: IG integration steps
        device: compute device
        eval_path: prefix for audio paths

    Returns:
        dict with EER curves and AUC metrics
    """
    if ratios is None:
        ratios = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    model.eval()
    ig = IntegratedGradients_ECAPA(model, n_steps=n_steps)

    # Collect scores for each ratio
    # For each sample pair, we have:
    #   label=1 (same speaker): target vs ref_same
    #   label=0 (diff speaker): target vs ref_diff
    # We compute cosine_sim scores at each ratio for both

    all_results = {
        'deletion': {r: {'same': [], 'diff': []} for r in ratios},
        'insertion': {r: {'same': [], 'diff': []} for r in ratios},
        'random_deletion': {r: {'same': [], 'diff': []} for r in ratios},
        'random_insertion': {r: {'same': [], 'diff': []} for r in ratios},
        'original': {'same': [], 'diff': []},
    }

    rng = np.random.default_rng(42)

    for pair in tqdm(sample_pairs, desc="Deletion/Insertion test"):
        target_path = os.path.join(eval_path, pair['target']) if eval_path else pair['target']
        ref_same_path = os.path.join(eval_path, pair['ref_same']) if eval_path else pair['ref_same']
        ref_diff_path = os.path.join(eval_path, pair['ref_diff']) if eval_path else pair['ref_diff']

        try:
            _, target_tensor = load_audio_as_tensor(target_path, device)
            _, ref_same_tensor = load_audio_as_tensor(ref_same_path, device)
            _, ref_diff_tensor = load_audio_as_tensor(ref_diff_path, device)
        except Exception as e:
            print(f"  Skipping {pair.get('label', '?')}: {e}")
            continue

        # Compute attribution
        if attribution_method == 'cosine_sim_diff':
            ig_pos = ig.generate(target_tensor, ref_tensor=ref_same_tensor,
                                 objective='cosine_sim', verify_convergence=False)
            ig_neg = ig.generate(target_tensor, ref_tensor=ref_diff_tensor,
                                 objective='cosine_sim', verify_convergence=False)
            attribution = ig_pos - ig_neg
        elif attribution_method == 'l2_norm':
            attribution = ig.generate(target_tensor, objective='l2_norm', verify_convergence=False)
        else:
            raise ValueError(f"Unknown attribution_method: {attribution_method}")

        # Run per-sample test
        result = deletion_insertion_test(
            model, target_tensor, ref_same_tensor, ref_diff_tensor,
            attribution, ratios=ratios, mode=mode, n_random=n_random, device=device
        )

        # Collect scores
        all_results['original']['same'].append(result['original_score_same'])
        all_results['original']['diff'].append(result['original_score_diff'])

        for r in ratios:
            all_results['deletion'][r]['same'].append(result['deletion_scores_same'][r])
            all_results['deletion'][r]['diff'].append(result['deletion_scores_diff'][r])
            all_results['insertion'][r]['same'].append(result['insertion_scores_same'][r])
            all_results['insertion'][r]['diff'].append(result['insertion_scores_diff'][r])
            all_results['random_deletion'][r]['same'].append(result['random_deletion_same'][r])
            all_results['random_deletion'][r]['diff'].append(result['random_deletion_diff'][r])
            all_results['random_insertion'][r]['same'].append(result['random_insertion_same'][r])
            all_results['random_insertion'][r]['diff'].append(result['random_insertion_diff'][r])

    # Compute EER at each ratio
    from tools import tuneThresholdfromScore

    def _compute_eer(same_scores, diff_scores):
        scores = same_scores + diff_scores
        labels = [1] * len(same_scores) + [0] * len(diff_scores)
        try:
            _, eer, _, _ = tuneThresholdfromScore(scores, labels, [1, 0.1])
            return eer
        except Exception:
            return float('nan')

    eer_curves = {
        'original': _compute_eer(all_results['original']['same'], all_results['original']['diff']),
        'deletion': {},
        'insertion': {},
        'random_deletion': {},
        'random_insertion': {},
    }

    for r in ratios:
        eer_curves['deletion'][r] = _compute_eer(
            all_results['deletion'][r]['same'], all_results['deletion'][r]['diff'])
        eer_curves['insertion'][r] = _compute_eer(
            all_results['insertion'][r]['same'], all_results['insertion'][r]['diff'])
        eer_curves['random_deletion'][r] = _compute_eer(
            all_results['random_deletion'][r]['same'], all_results['random_deletion'][r]['diff'])
        eer_curves['random_insertion'][r] = _compute_eer(
            all_results['random_insertion'][r]['same'], all_results['random_insertion'][r]['diff'])

    # Compute AUC (area under EER curve)
    # For deletion: as ratio increases, EER should increase → we measure how fast
    # For insertion: as ratio increases, EER should decrease → we measure how fast
    # AUC is computed on the "relative EER change" curve

    def _compute_auc(ratios_sorted, eer_values, original_eer, direction='deletion'):
        """
        Compute AUC of the normalized EER curve.
        For deletion: normalize by original_eer, higher AUC = better (more EER increase per deletion)
        For insertion: normalize by original_eer, lower AUC = better (faster EER recovery)
        """
        if np.isnan(original_eer) or original_eer < 1e-6:
            return float('nan')

        # Normalized EER: eer / original_eer
        norm_eers = [eer_values.get(r, float('nan')) / original_eer for r in ratios_sorted]
        valid = [(r, e) for r, e in zip(ratios_sorted, norm_eers) if not np.isnan(e)]
        if len(valid) < 2:
            return float('nan')

        rs, es = zip(*valid)
        # AUC using trapezoidal rule
        auc = np.trapz(es, rs)
        return auc

    ratios_sorted = sorted(ratios)

    deletion_auc = _compute_auc(ratios_sorted, eer_curves['deletion'],
                                eer_curves['original'], direction='deletion')
    insertion_auc = _compute_auc(ratios_sorted, eer_curves['insertion'],
                                 eer_curves['original'], direction='insertion')
    random_deletion_auc = _compute_auc(ratios_sorted, eer_curves['random_deletion'],
                                       eer_curves['original'], direction='deletion')
    random_insertion_auc = _compute_auc(ratios_sorted, eer_curves['random_insertion'],
                                        eer_curves['original'], direction='insertion')

    return {
        'ratios': ratios_sorted,
        'eer_curves': eer_curves,
        'deletion_auc': deletion_auc,
        'insertion_auc': insertion_auc,
        'random_deletion_auc': random_deletion_auc,
        'random_insertion_auc': random_insertion_auc,
        'all_scores': all_results,
    }


# ──────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────

def plot_reliability_curves(results: Dict, save_path: str,
                            model_name: str = '', attribution_method: str = '',
                            mode: str = 'freq_time'):
    """
    Plot Deletion and Insertion EER curves.

    Left: Deletion curve (x=ratio deleted, y=EER)
    Right: Insertion curve (x=ratio inserted, y=EER)

    Each plot shows: attribution-based, random baseline, and original EER.
    """
    ratios = results['ratios']
    eer = results['eer_curves']
    original_eer = eer['original']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Deletion ──
    del_eers = [eer['deletion'].get(r, float('nan')) for r in ratios]
    rand_del_eers = [eer['random_deletion'].get(r, float('nan')) for r in ratios]

    ax1.plot(ratios, del_eers, 'o-', color='#e74c3c', linewidth=2, markersize=6,
             label=f'{attribution_method} attribution')
    ax1.plot(ratios, rand_del_eers, 's--', color='#95a5a6', linewidth=1.5, markersize=5,
             label='Random baseline')
    ax1.axhline(y=original_eer, color='#2ecc71', linestyle=':', linewidth=1.5,
                label=f'Original EER ({original_eer:.2f}%)')
    ax1.set_xlabel('Deletion Ratio', fontsize=12)
    ax1.set_ylabel('EER (%)', fontsize=12)
    ax1.set_title(f'Deletion Test — {model_name}\n(mode={mode})', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, max(ratios) + 0.02)
    ax1.set_ylim(0, max(max(del_eers + rand_del_eers + [original_eer]) * 1.2, 1))
    ax1.grid(True, alpha=0.3)

    # Add AUC annotation
    auc_del = results.get('deletion_auc', float('nan'))
    auc_rand_del = results.get('random_deletion_auc', float('nan'))
    ax1.text(0.05, 0.95, f'AUC: {auc_del:.3f}\nRandom AUC: {auc_rand_del:.3f}',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ── Insertion ──
    ins_eers = [eer['insertion'].get(r, float('nan')) for r in ratios]
    rand_ins_eers = [eer['random_insertion'].get(r, float('nan')) for r in ratios]

    ax2.plot(ratios, ins_eers, 'o-', color='#3498db', linewidth=2, markersize=6,
             label=f'{attribution_method} attribution')
    ax2.plot(ratios, rand_ins_eers, 's--', color='#95a5a6', linewidth=1.5, markersize=5,
             label='Random baseline')
    ax2.axhline(y=original_eer, color='#2ecc71', linestyle=':', linewidth=1.5,
                label=f'Original EER ({original_eer:.2f}%)')
    ax2.set_xlabel('Insertion Ratio', fontsize=12)
    ax2.set_ylabel('EER (%)', fontsize=12)
    ax2.set_title(f'Insertion Test — {model_name}\n(mode={mode})', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, max(ratios) + 0.02)
    ax2.set_ylim(0, max(max(ins_eers + rand_ins_eers + [original_eer]) * 1.2, 1))
    ax2.grid(True, alpha=0.3)

    auc_ins = results.get('insertion_auc', float('nan'))
    auc_rand_ins = results.get('random_insertion_auc', float('nan'))
    ax2.text(0.05, 0.95, f'AUC: {auc_ins:.3f}\nRandom AUC: {auc_rand_ins:.3f}',
             transform=ax2.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Attribution Reliability: Deletion/Insertion Test\n'
                 f'Model: {model_name} | Method: {attribution_method} | Mode: {mode}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_multi_model_comparison(all_results: Dict[str, Dict], save_path: str,
                                attribution_method: str = '', mode: str = 'freq_time'):
    """
    Plot Deletion/Insertion curves comparing multiple models.

    Args:
        all_results: {model_name: batch_deletion_insertion_test result dict}
    """
    n_models = len(all_results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#e74c3c', '#3498db', '#9b59b6', '#e67e22', '#1abc9c']

    for idx, (model_name, result) in enumerate(all_results.items()):
        ratios = result['ratios']
        eer = result['eer_curves']
        color = colors[idx % len(colors)]

        del_eers = [eer['deletion'].get(r, float('nan')) for r in ratios]
        ins_eers = [eer['insertion'].get(r, float('nan')) for r in ratios]

        ax1.plot(ratios, del_eers, 'o-', color=color, linewidth=2, markersize=5,
                 label=f'{model_name} (AUC={result["deletion_auc"]:.3f})')
        ax2.plot(ratios, ins_eers, 'o-', color=color, linewidth=2, markersize=5,
                 label=f'{model_name} (AUC={result["insertion_auc"]:.3f})')

    # Random baseline (use first model's)
    first_result = list(all_results.values())[0]
    ratios = first_result['ratios']
    eer = first_result['eer_curves']
    original_eer = eer['original']
    rand_del = [eer['random_deletion'].get(r, float('nan')) for r in ratios]
    rand_ins = [eer['random_insertion'].get(r, float('nan')) for r in ratios]

    ax1.plot(ratios, rand_del, 's--', color='#95a5a6', linewidth=1.5, markersize=4,
             label='Random baseline')
    ax1.axhline(y=original_eer, color='#2ecc71', linestyle=':', linewidth=1.5,
                label=f'Original EER ({original_eer:.2f}%)')

    ax2.plot(ratios, rand_ins, 's--', color='#95a5a6', linewidth=1.5, markersize=4,
             label='Random baseline')
    ax2.axhline(y=original_eer, color='#2ecc71', linestyle=':', linewidth=1.5,
                label=f'Original EER ({original_eer:.2f}%)')

    ax1.set_xlabel('Deletion Ratio', fontsize=12)
    ax1.set_ylabel('EER (%)', fontsize=12)
    ax1.set_title(f'Deletion Test — {attribution_method} (mode={mode})', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Insertion Ratio', fontsize=12)
    ax2.set_ylabel('EER (%)', fontsize=12)
    ax2.set_title(f'Insertion Test — {attribution_method} (mode={mode})', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Attribution Reliability: Multi-Model Deletion/Insertion Comparison',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_method_comparison(cosine_results: Dict, l2_results: Dict, save_path: str,
                           model_name: str = '', mode: str = 'freq_time'):
    """
    Plot Deletion/Insertion curves comparing cosine_sim_diff vs l2_norm
    for a single model.

    Args:
        cosine_results: batch result for cosine_sim_diff
        l2_results: batch result for l2_norm
    """
    ratios = cosine_results['ratios']
    eer_cos = cosine_results['eer_curves']
    eer_l2 = l2_results['eer_curves']
    original_eer = eer_cos['original']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cos_del = [eer_cos['deletion'].get(r, float('nan')) for r in ratios]
    l2_del = [eer_l2['deletion'].get(r, float('nan')) for r in ratios]
    rand_del = [eer_cos['random_deletion'].get(r, float('nan')) for r in ratios]

    cos_ins = [eer_cos['insertion'].get(r, float('nan')) for r in ratios]
    l2_ins = [eer_l2['insertion'].get(r, float('nan')) for r in ratios]
    rand_ins = [eer_cos['random_insertion'].get(r, float('nan')) for r in ratios]

    ax1.plot(ratios, cos_del, 'o-', color='#e74c3c', linewidth=2, markersize=6,
             label=f'Ours: cos_sim_diff (AUC={cosine_results["deletion_auc"]:.3f})')
    ax1.plot(ratios, l2_del, '^-', color='#3498db', linewidth=2, markersize=6,
             label=f'L2-norm (AUC={l2_results["deletion_auc"]:.3f})')
    ax1.plot(ratios, rand_del, 's--', color='#95a5a6', linewidth=1.5, markersize=5,
             label='Random baseline')
    ax1.axhline(y=original_eer, color='#2ecc71', linestyle=':', linewidth=1.5,
                label=f'Original EER ({original_eer:.2f}%)')
    ax1.set_xlabel('Deletion Ratio', fontsize=12)
    ax1.set_ylabel('EER (%)', fontsize=12)
    ax1.set_title(f'Deletion Test — {model_name}', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ratios, cos_ins, 'o-', color='#e74c3c', linewidth=2, markersize=6,
             label=f'Ours: cos_sim_diff (AUC={cosine_results["insertion_auc"]:.3f})')
    ax2.plot(ratios, l2_ins, '^-', color='#3498db', linewidth=2, markersize=6,
             label=f'L2-norm (AUC={l2_results["insertion_auc"]:.3f})')
    ax2.plot(ratios, rand_ins, 's--', color='#95a5a6', linewidth=1.5, markersize=5,
             label='Random baseline')
    ax2.axhline(y=original_eer, color='#2ecc71', linestyle=':', linewidth=1.5,
                label=f'Original EER ({original_eer:.2f}%)')
    ax2.set_xlabel('Insertion Ratio', fontsize=12)
    ax2.set_ylabel('EER (%)', fontsize=12)
    ax2.set_title(f'Insertion Test — {model_name}', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Method Comparison: Ours (cos_sim_diff) vs L2-norm — {model_name}\n'
                 f'(mode={mode})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
