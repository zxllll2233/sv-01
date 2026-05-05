import os
import glob
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import soundfile as sf
from typing import List, Dict, Union
import time

from .integrated_gradients import IntegratedGradients_ECAPA

FREQ_BANDS = {
    'F0\n(80-300Hz)': (0, 5),
    'F1\n(300-1kHz)': (5, 17),
    'F2\n(1k-2.5kHz)': (17, 35),
    'F3\n(2.5k-4kHz)': (35, 52),
    'High\n(4k-7.6kHz)': (52, 80),
}

_TARGET_LENGTH = 300 * 160 + 240


def load_audio_as_tensor(audio_path, device='cuda'):
    audio, sr = sf.read(audio_path)
    length = _TARGET_LENGTH
    if audio.shape[0] <= length:
        shortage = length - audio.shape[0]
        audio = np.pad(audio, (0, shortage), 'wrap')
    else:
        audio = audio[:length]
    return audio, torch.FloatTensor(audio).unsqueeze(0).to(device)


def compute_fbank(model, audio_tensor):
    with torch.no_grad():
        fbank = model.torchfbank(audio_tensor) + 1e-6
        fbank = fbank.log()
        fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
        return fbank.squeeze().cpu().numpy()


def add_freq_band_labels(ax):
    for band_name, (lo, hi) in FREQ_BANDS.items():
        mid = (lo + hi) / 2
        ax.axhline(y=hi, color='white', linewidth=0.5, alpha=0.6, linestyle='--')
        ax.text(ax.get_xlim()[1] * 1.02, mid, band_name,
                fontsize=6, va='center', ha='left', color='#333333')


def add_colorbar(ax, im=None, visible=True):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if visible and im is not None:
        plt.colorbar(im, cax=cax)
    else:
        cax.axis('off')
    return cax


def compute_band_energy(ig):
    if ig.ndim == 3 and ig.shape[0] == 1:
        ig = ig.squeeze(0)
    total = np.abs(ig).sum() + 1e-10
    energies = {}
    for band_name, (lo, hi) in FREQ_BANDS.items():
        short_name = band_name.split('\n')[0]
        energies[short_name] = np.abs(ig[lo:hi, :]).sum() / total
    return energies


def plot_voiceprint_highlight(ax, fbank, ig_diff, title='',
                               threshold_percentile=70,
                               fbank_cmap='gray_r', fbank_alpha=0.6):
    if ig_diff.ndim == 3 and ig_diff.shape[0] == 1:
        ig_diff = ig_diff.squeeze(0)

    T = fbank.shape[1]
    F = fbank.shape[0]
    extent = [0, T, 0, F]

    ax.imshow(fbank, origin='lower', aspect='auto', cmap=fbank_cmap,
              alpha=fbank_alpha, extent=extent)

    limit = max(np.percentile(np.abs(ig_diff), 99), 1e-8)
    ig_norm = ig_diff / limit

    pos_threshold = np.percentile(ig_norm[ig_norm > 0], max(0, 100 - threshold_percentile)) if (ig_norm > 0).any() else 0.5

    rgba = np.zeros((*ig_norm.shape, 4))

    pos_mask = ig_norm > pos_threshold
    pos_strength = np.clip((ig_norm - pos_threshold) / (1 - pos_threshold + 1e-8), 0, 1)
    rgba[pos_mask, 0] = 1.0
    rgba[pos_mask, 1] = 0.3
    rgba[pos_mask, 2] = 0.6
    rgba[pos_mask, 3] = pos_strength[pos_mask] * 0.9

    ax.imshow(rgba, origin='lower', aspect='auto', interpolation='bicubic', extent=extent)

    add_freq_band_labels(ax)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(1.0, 0.3, 0.6), alpha=0.8, label='Voiceprint (+)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
              framealpha=0.7, handlelength=1)

    ax.set_title(title, fontsize=9)

class ECAPAAttributionAnalyzer:
    def __init__(self, models_dict: Dict[str, torch.nn.Module], C=512, n_steps=50, device='cuda', musan_path=None, baseline_computer=None):
        self.models = models_dict
        self.device = device
        self.target_length = _TARGET_LENGTH
        self.baseline_computer = baseline_computer
        
        # Initialize IG for each model
        self.igs = {}
        for name, model in self.models.items():
            self.igs[name] = IntegratedGradients_ECAPA(model, n_steps=n_steps)
            
        # Noise Augmentation Setup
        self.musan_path = musan_path
        self.noise_files = []
        if self.musan_path and os.path.exists(self.musan_path):
            print(f"[Attribution] Scanning noise files in {self.musan_path}...")
            self.noise_files = glob.glob(os.path.join(self.musan_path, '*/*/*.wav'))
            print(f"[Attribution] Found {len(self.noise_files)} noise files.")
        
        # SNR Configuration (Reference from dataLoader.py)
        self.noisesnr = {'noise':[0,15], 'speech':[13,20], 'music':[5,15]}

    def _load_audio_as_tensor(self, audio_path):
        return load_audio_as_tensor(audio_path, self.device)

    def _get_noise_audio(self):
        """Randomly select a noise file and load it formatted"""
        if not self.noise_files:
            return None, None
            
        noise_path = random.choice(self.noise_files)
        noise_wav, _ = self._load_audio_as_tensor(noise_path)
        return noise_wav, noise_path

    def _augment_audio(self, clean_wav, noise_wav, noise_path, snr=None):
        if snr is not None:
            target_snr = snr
        else:
            try:
                parts = noise_path.split(os.sep)
                category = parts[-3]
                if category not in self.noisesnr:
                    category = 'noise'
            except:
                category = 'noise'
            snr_range = self.noisesnr.get(category, [0, 15])
            target_snr = random.uniform(snr_range[0], snr_range[1])

        clean_db = 10 * np.log10(np.mean(clean_wav ** 2) + 1e-4)
        noise_db = 10 * np.log10(np.mean(noise_wav ** 2) + 1e-4)
        
        scale = np.sqrt(10 ** ((clean_db - noise_db - target_snr) / 10))
        noise_scaled = scale * noise_wav
        noisy_wav = clean_wav + noise_scaled
        
        return noisy_wav, noise_scaled, target_snr

    def analyze_waveform(self, waveform: np.ndarray) -> Dict:
        """Analyze a specific waveform (numpy array)"""
        audio_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
        model_results = {}
        
        for name, model in self.models.items():
            model.eval()
            ig_analyzer = self.igs[name]
            
            # FBank
            with torch.no_grad():
                fbank = model.torchfbank(audio_tensor) + 1e-6
                fbank = fbank.log()
                fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
                fbank = fbank.squeeze().cpu().numpy()
                
            # IG
            ig_map = ig_analyzer.generate(audio_tensor, objective='l2_norm', verify_convergence=False)
            
            model_results[name] = {
                'fbank': fbank,
                'ig': ig_map
            }
        
        return model_results

    def analyze_paired(self, audio_tensor, ref_same_tensor, ref_diff_tensor,
                       baseline_type='zero', objective='cosine_sim',
                       speaker_audio_paths=None, all_audio_paths=None,
                       exclude_speaker_paths=None):
        """
        配对归因分析：正例（同说话人）+ 反例（不同说话人）+ 差值

        Args:
            audio_tensor: 目标音频 [1, samples]
            ref_same_tensor: 同一说话人参考音频 [1, samples]
            ref_diff_tensor: 不同说话人参考音频 [1, samples]
            baseline_type: 'zero' | 'global_mean' | 'speaker_mean' | 'cross_speaker_mean'
            objective: 归因目标 ('cosine_sim' 或 'l2_norm')

        Returns:
            {
                'positive': {model_name: {'fbank': ..., 'ig': ...}},
                'negative': {model_name: {'fbank': ..., 'ig': ...}},
                'difference': {model_name: {'ig_diff': ...}},
                'baseline_type': baseline_type
            }
        """
        with torch.no_grad():
            first_model = list(self.models.values())[0]
            fbank = first_model.torchfbank(audio_tensor) + 1e-6
            fbank = fbank.log()
            fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
            fbank = fbank.squeeze().cpu().numpy()

            ref_same_fbank = first_model.torchfbank(ref_same_tensor) + 1e-6
            ref_same_fbank = ref_same_fbank.log()
            ref_same_fbank = ref_same_fbank - torch.mean(ref_same_fbank, dim=-1, keepdim=True)
            ref_same_fbank = ref_same_fbank.squeeze().cpu().numpy()

            ref_diff_fbank = first_model.torchfbank(ref_diff_tensor) + 1e-6
            ref_diff_fbank = ref_diff_fbank.log()
            ref_diff_fbank = ref_diff_fbank - torch.mean(ref_diff_fbank, dim=-1, keepdim=True)
            ref_diff_fbank = ref_diff_fbank.squeeze().cpu().numpy()

        if baseline_type != 'zero' and self.baseline_computer is not None:
            baseline = self.baseline_computer.get_baseline(
                baseline_type, input_fbank_shape=[1, 80, fbank.shape[-1]],
                speaker_audio_paths=speaker_audio_paths,
                all_audio_paths=all_audio_paths,
                exclude_speaker_paths=exclude_speaker_paths
            )
        else:
            baseline = None

        positive_results = {}
        negative_results = {}
        difference_results = {}

        for name, model in self.models.items():
            model.eval()
            ig_analyzer = self.igs[name]

            ig_positive = ig_analyzer.generate(
                audio_tensor,
                ref_tensor=ref_same_tensor,
                baseline=baseline,
                objective=objective,
                verify_convergence=True
            )

            ig_negative = ig_analyzer.generate(
                audio_tensor,
                ref_tensor=ref_diff_tensor,
                baseline=baseline,
                objective=objective,
                verify_convergence=True
            )

            ig_diff = ig_positive - ig_negative

            positive_results[name] = {'fbank': fbank, 'ig': ig_positive}
            negative_results[name] = {'fbank': fbank, 'ig': ig_negative}
            difference_results[name] = {'ig_diff': ig_diff}

        return {
            'positive': positive_results,
            'negative': negative_results,
            'difference': difference_results,
            'fbank_target': fbank,
            'fbank_ref_same': ref_same_fbank,
            'fbank_ref_diff': ref_diff_fbank,
            'baseline_type': baseline_type
        }

    # Mel bin → 语音学频段映射 (ECAPA-TDNN: n_mels=80, f_max=7600, f_min=20, sr=16000)
    FREQ_BANDS = FREQ_BANDS

    def _add_freq_band_labels(self, ax):
        add_freq_band_labels(ax)

    def _add_colorbar(self, ax, im=None, visible=True):
        return add_colorbar(ax, im, visible)

    def _compute_band_energy(self, ig):
        return compute_band_energy(ig)

    def _plot_voiceprint_highlight(self, ax, fbank, ig_diff, title='',
                                   threshold_percentile=70,
                                   fbank_cmap='gray_r', fbank_alpha=0.6):
        plot_voiceprint_highlight(ax, fbank, ig_diff, title,
                                  threshold_percentile, fbank_cmap, fbank_alpha)

    def compute_model_attribution(self, target_tensor, ref_same_tensor, ref_diff_tensor,
                                   noisy_tensor=None,
                                   baseline_type='zero',
                                   speaker_audio_paths=None, all_audio_paths=None,
                                   exclude_speaker_paths=None):
        results = {}

        if baseline_type != 'zero' and self.baseline_computer is not None:
            with torch.no_grad():
                fb = list(self.models.values())[0].torchfbank(target_tensor)
            baseline = self.baseline_computer.get_baseline(
                baseline_type, input_fbank_shape=list(fb.shape),
                speaker_audio_paths=speaker_audio_paths,
                all_audio_paths=all_audio_paths,
                exclude_speaker_paths=exclude_speaker_paths
            )
        else:
            baseline = None

        for name, model in self.models.items():
            model.eval()
            ig = self.igs[name]

            l2_map = ig.generate(target_tensor, objective='l2_norm', verify_convergence=False)

            cos_pos = ig.generate(target_tensor, ref_tensor=ref_same_tensor,
                                   baseline=baseline, objective='cosine_sim', verify_convergence=True)
            cos_neg = ig.generate(target_tensor, ref_tensor=ref_diff_tensor,
                                   baseline=baseline, objective='cosine_sim', verify_convergence=True)
            cos_diff = cos_pos - cos_neg

            model_result = {
                'l2_norm': l2_map,
                'cosine_sim_diff': cos_diff,
            }

            if noisy_tensor is not None:
                l2_noisy_map = ig.generate(noisy_tensor, objective='l2_norm', verify_convergence=False)
                model_result['l2_norm_noisy'] = l2_noisy_map

                noisy_pos = ig.generate(noisy_tensor, ref_tensor=ref_same_tensor,
                                         baseline=baseline, objective='cosine_sim', verify_convergence=True)
                noisy_neg = ig.generate(noisy_tensor, ref_tensor=ref_diff_tensor,
                                         baseline=baseline, objective='cosine_sim', verify_convergence=True)
                model_result['cosine_sim_noisy_diff'] = noisy_pos - noisy_neg

            results[name] = model_result

        return results

def visualize_attribution_6row(save_path, model_names,
                                fbank_clean, fbank_noise, fbank_noisy,
                                l2_attrs, l2_noisy_attrs, cos_clean_attrs, cos_noisy_attrs,
                                audio_label="", noise_info="", baseline_type="zero"):
    num_models = len(model_names)
    n_rows = 6
    fbank_cols = 3
    num_cols = max(num_models, fbank_cols)
    row_h = 3.0

    fig, axes = plt.subplots(n_rows, num_cols,
                             figsize=(5 * num_cols, row_h * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    plt.rcParams.update({'font.size': 9})

    T = fbank_clean.shape[1]
    F = fbank_clean.shape[0]
    extent = [0, T, 0, F]

    def _plot_fbank(ax, fb, title, cmap='viridis'):
        im = ax.imshow(fb, origin='lower', aspect='auto', cmap=cmap, extent=extent)
        ax.set_title(title, fontsize=9)
        add_freq_band_labels(ax)
        add_colorbar(ax, im, visible=True)

    # Row 0: Clean | Noise | Noisy
    _plot_fbank(axes[0, 0], fbank_clean, f"Clean Target\n{audio_label}")
    _plot_fbank(axes[0, 1], fbank_noise, f"Noise\n{noise_info}")
    _plot_fbank(axes[0, 2], fbank_noisy, f"Noisy Target\n{noise_info}")
    for ci in range(fbank_cols, num_cols):
        axes[0, ci].axis('off')

    # Row 1: Model label separator
    for ci in range(num_cols):
        ax = axes[1, ci]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if ci < num_models:
            ax.text(0.5, 0.5, f"▼ {model_names[ci]} ▼",
                    fontsize=11, fontweight='bold', ha='center', va='center',
                    transform=ax.transAxes, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8e8e8', edgecolor='#999999'))

    # Row 2: Ours: cos_sim (Clean) per model
    for m_idx, name in enumerate(model_names):
        ax = axes[2, m_idx]
        plot_voiceprint_highlight(ax, fbank_clean, cos_clean_attrs[name],
                                  f"Ours: cos_sim (Clean)\n{name}")
    for m_idx in range(num_models, num_cols):
        axes[2, m_idx].axis('off')

    # Row 3: Ours: cos_sim (Noisy) per model
    for m_idx, name in enumerate(model_names):
        ax = axes[3, m_idx]
        plot_voiceprint_highlight(ax, fbank_noisy, cos_noisy_attrs[name],
                                  f"Ours: cos_sim (Noisy)\n{name}")
    for m_idx in range(num_models, num_cols):
        axes[3, m_idx].axis('off')

    # Row 4: L2-norm (Clean) per model
    for m_idx, name in enumerate(model_names):
        ax = axes[4, m_idx]
        plot_voiceprint_highlight(ax, fbank_clean, l2_attrs[name],
                                  f"L2-norm (Clean)\n{name}")
    for m_idx in range(num_models, num_cols):
        axes[4, m_idx].axis('off')

    # Row 5: L2-norm (Noisy) per model
    for m_idx, name in enumerate(model_names):
        ax = axes[5, m_idx]
        plot_voiceprint_highlight(ax, fbank_noisy, l2_noisy_attrs[name],
                                  f"L2-norm (Noisy)\n{name}")
    for m_idx in range(num_models, num_cols):
        axes[5, m_idx].axis('off')

    noise_suffix = f" | {noise_info}" if noise_info else ""
    fig.suptitle(f"Voiceprint Attribution (Baseline: {baseline_type}{noise_suffix})",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    def analyze_and_save_paired(self, sample_pairs, save_dir, baseline_type='zero',
                                objective='cosine_sim', base_path=None,
                                all_audio_paths=None, add_noise=False, snr=None):
        os.makedirs(save_dir, exist_ok=True)

        for i, pair in enumerate(sample_pairs):
            target_path = os.path.join(base_path, pair['target']) if base_path else pair['target']
            ref_same_path = os.path.join(base_path, pair['ref_same']) if base_path else pair['ref_same']
            ref_diff_path = os.path.join(base_path, pair['ref_diff']) if base_path else pair['ref_diff']
            label = pair.get('label', f'sample_{i}')

            print(f"[Paired Attribution] Analyzing {label}...")

            try:
                target_wav, target_tensor = self._load_audio_as_tensor(target_path)
                _, ref_same_tensor = self._load_audio_as_tensor(ref_same_path)
                _, ref_diff_tensor = self._load_audio_as_tensor(ref_diff_path)

                noisy_tensor = None
                noise_wav = None
                noise_info = ""

                if add_noise and self.noise_files:
                    noise_wav, noise_path = self._get_noise_audio()
                    if noise_wav is not None:
                        noisy_wav, _, actual_snr = self._augment_audio(
                            target_wav, noise_wav, noise_path, snr=snr
                        )
                        noisy_tensor = torch.FloatTensor(noisy_wav).unsqueeze(0).to(self.device)
                        noise_info = f"{os.path.basename(noise_path)} (SNR={actual_snr:.1f}dB)"

                speaker_audio_paths = None
                exclude_speaker_paths = None
                if baseline_type != 'zero' and all_audio_paths:
                    target_dir = os.path.dirname(target_path)
                    speaker_audio_paths = [
                        p for p in all_audio_paths
                        if os.path.dirname(p) == target_dir
                    ]
                    if baseline_type == 'cross_speaker_mean':
                        exclude_speaker_paths = speaker_audio_paths

                first_model = list(self.models.values())[0]
                fbank_clean = compute_fbank(first_model, target_tensor)
                fbank_noisy = compute_fbank(first_model, noisy_tensor) if noisy_tensor is not None else fbank_clean

                fbank_noise = None
                if noise_wav is not None:
                    noise_tensor = torch.FloatTensor(noise_wav).unsqueeze(0).to(self.device)
                    fbank_noise = compute_fbank(first_model, noise_tensor)

                model_attrs = self.compute_model_attribution(
                    target_tensor, ref_same_tensor, ref_diff_tensor,
                    noisy_tensor=noisy_tensor,
                    baseline_type=baseline_type,
                    speaker_audio_paths=speaker_audio_paths,
                    all_audio_paths=all_audio_paths,
                    exclude_speaker_paths=exclude_speaker_paths
                )

                save_path = os.path.join(save_dir, f"{label}_paired_attribution.png")

                model_names = list(model_attrs.keys())
                l2_attrs = {n: model_attrs[n]['l2_norm'] for n in model_names}
                l2_noisy_attrs = {n: model_attrs[n].get('l2_norm_noisy', model_attrs[n]['l2_norm']) for n in model_names}
                cos_clean_attrs = {n: model_attrs[n]['cosine_sim_diff'] for n in model_names}
                cos_noisy_attrs = {n: model_attrs[n].get('cosine_sim_noisy_diff', model_attrs[n]['cosine_sim_diff']) for n in model_names}

                visualize_attribution_6row(
                    save_path, model_names,
                    fbank_clean, fbank_noise if fbank_noise is not None else fbank_noisy,
                    fbank_noisy, l2_attrs, l2_noisy_attrs, cos_clean_attrs, cos_noisy_attrs,
                    audio_label=os.path.basename(target_path),
                    noise_info=noise_info,
                    baseline_type=baseline_type
                )
                print(f"[Paired Attribution] Saved: {save_path}")

            except Exception as e:
                print(f"[Paired Attribution] Error analyzing {label}: {str(e)}")
                import traceback
                traceback.print_exc()

    def analyze(self, audio_path: str) -> Dict:
        """Legacy method for single file analysis (Clean only)"""
        audio, _ = self._load_audio_as_tensor(audio_path)
        model_results = self.analyze_waveform(audio)
        return {
            'waveform': audio,
            'path': audio_path,
            'model_results': model_results
        }

    def _add_colorbar(self, ax, im=None, visible=True):
        """
        Helper function to add a colorbar that doesn't distort the aspect ratio
        of the main plot, ensuring alignment between subplots.
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if visible and im is not None:
            plt.colorbar(im, cax=cax)
        else:
            cax.axis('off')
        return cax

    def visualize_ig_comparison(self, full_results: Dict, save_path: str):
        """
        Visualize 6 rows x N columns:
        Rows 1-2: Clean (Wave, FBank, Model IGs, Overlays)
        Rows 3-4: Noise (Wave, FBank, Model IGs, Overlays)
        Rows 5-6: Noisy (Wave, FBank, Model IGs, Overlays)
        """
        model_names = list(self.models.keys())
        num_models = len(model_names)
        num_cols = num_models + 1
        
        # 6行: 3 groups * 2 rows/group
        # figsize: width = 6*num_cols, height = 4 * 6 = 24? Too tall.
        # Let's adjust height. 24 is fine for high res.
        fig, axes = plt.subplots(6, num_cols, figsize=(6 * num_cols, 20), sharex=True, sharey=False)
        plt.rcParams.update({'font.size': 10})
        
        groups = [
            ('noise', 'Pure Noise', f"\n({full_results['meta']['noise_short_path']})"),
            ('clean', 'Clean Speech', ""),
            ('noisy', 'Noisy Speech', "")
        ]
        
        for g_idx, (key, label, suffix) in enumerate(groups):
            row_start = g_idx * 2
            data = full_results[key]
            waveform = data['waveform']
            model_results = data['model_results']
            
            # --- Column 1: Input ---
            # Waveform
            ax_wave = axes[row_start, 0]
            time_axis = np.arange(len(waveform)) / 160.0
            ax_wave.plot(time_axis, waveform, color='black', linewidth=0.5)
            ax_wave.set_ylabel(f"{label}\nAmplitude")
            ax_wave.set_title(f"{label} Waveform{suffix}")
            
            # FBank (Use first model's fbank)
            first_fbank = model_results[model_names[0]]['fbank']
            ax_fbank = axes[row_start + 1, 0]
            im_fbank = ax_fbank.imshow(first_fbank, origin='lower', aspect='auto', cmap='jet', extent=[0, first_fbank.shape[1], 0, first_fbank.shape[0]])
            ax_fbank.set_ylabel("Mel Filter")
            ax_fbank.set_title("FBank Features")
            self._add_colorbar(ax_fbank, im_fbank, visible=True)
            
            # Set X limit based on fbank
            ax_wave.set_xlim(0, first_fbank.shape[1])
            self._add_colorbar(ax_wave, visible=False)
            
            # --- Columns 2..N: Models ---
            for m_idx, name in enumerate(model_names):
                col_idx = m_idx + 1
                m_res = model_results[name]
                fbank = m_res['fbank']
                ig = m_res['ig']
                
                if ig.ndim == 3 and ig.shape[0] == 1: ig = ig.squeeze(0)
                
                # Row 0 of group: IG
                ax_ig = axes[row_start, col_idx]
                limit = max(np.percentile(np.abs(ig), 99), 1e-8)
                im_ig = ax_ig.imshow(ig, origin='lower', aspect='auto', cmap='coolwarm',
                                     vmin=-limit, vmax=limit, interpolation='bicubic',
                                     extent=[0, fbank.shape[1], 0, fbank.shape[0]])
                ax_ig.set_title(f"{name}\nIG ({label})")
                ax_ig.set_yticklabels([])
                self._add_colorbar(ax_ig, im_ig, visible=True)
                
                # Row 1 of group: Overlay
                ax_over = axes[row_start + 1, col_idx]
                
                # Normalization based on Signed values (Positive & Negative)
                # Use the same limit as the IG plot above for consistency
                limit_abs = max(np.percentile(np.abs(ig), 99), 1e-8)
                
                # Normalize signed IG to [-1, 1]
                ig_norm_signed = np.clip(ig / limit_abs, -1, 1)
                
                # Calculate Magnitude for Alpha channel (importance strength)
                ig_norm_magnitude = np.abs(ig_norm_signed)
                
                # Gamma correction to boost visibility of mid-range importance
                # Lower gamma (0.5) makes smaller values more visible
                ig_norm_magnitude = np.power(ig_norm_magnitude, 0.5)
                
                # Background: FBank (Gray_r: White=Low, Black=High)
                # alpha=0.6 for clearer background context
                ax_over.imshow(fbank, origin='lower', aspect='auto', cmap='gray_r', alpha=0.6, extent=[0, fbank.shape[1], 0, fbank.shape[0]])
                
                # Overlay: Coolwarm (Blue=Negative, Red=Positive)
                cmap = plt.get_cmap('coolwarm')
                # Map [-1, 1] to [0, 1] for colormap
                overlay_rgba = cmap((ig_norm_signed + 1) / 2)
                
                # Set Alpha channel based on Magnitude
                # We want transparent for 0 (unimportant), Opaque for |1| (highly positive or negative)
                # Scale alpha max to 0.9 for high contrast
                overlay_rgba[..., 3] = ig_norm_magnitude * 0.9
                
                im_over = ax_over.imshow(overlay_rgba, origin='lower', aspect='auto', interpolation='bicubic', extent=[0, fbank.shape[1], 0, fbank.shape[0]])
                
                ax_over.set_title("Importance Overlay")
                ax_over.set_yticklabels([])
                
                # Custom Colorbar for Coolwarm (Signed)
                sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-limit_abs, vmax=limit_abs))
                sm.set_array([])
                self._add_colorbar(ax_over, sm, visible=True)
                
                if row_start + 1 == 5: # Last row
                    ax_over.set_xlabel("Time (Frames)")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def analyze_and_save(self, audio_paths: List[str], save_dir: str, prefix: str = ""):
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Select ONE noise file for the entire batch
        batch_noise_wav, batch_noise_path = self._get_noise_audio()
        
        if batch_noise_wav is None:
             print("[Attribution] No noise found! Will use clean as dummy noise for all.")
             batch_noise_path = "No_Noise_Found"
        else:
             print(f"[Attribution] Selected fixed noise for this batch: {batch_noise_path}")

        for i, path in enumerate(audio_paths):
            if not os.path.exists(path):
                print(f"[Attribution] Warning: Sample file {path} not found.")
                continue
                
            try:
                # 1. Load Clean Audio
                clean_wav, _ = self._load_audio_as_tensor(path)
                
                # 2. Augment (using the pre-selected batch noise)
                if batch_noise_wav is None:
                    current_noise_wav = np.zeros_like(clean_wav) + 1e-6
                    noisy_wav = clean_wav
                    category = "none"
                else:
                    current_noise_wav = batch_noise_wav
                    noisy_wav, _, category = self._augment_audio(clean_wav, batch_noise_wav, batch_noise_path)

                # 3. Analyze all 3
                print(f"[Attribution] Analyzing {os.path.basename(path)} with noise {os.path.basename(batch_noise_path)}...")
                
                res_clean = {'waveform': clean_wav, 'model_results': self.analyze_waveform(clean_wav)}
                res_noise = {'waveform': current_noise_wav, 'model_results': self.analyze_waveform(current_noise_wav)}
                res_noisy = {'waveform': noisy_wav, 'model_results': self.analyze_waveform(noisy_wav)}
                
                # 4. Prepare Metadata
                norm_path = os.path.normpath(path)
                parts = norm_path.split(os.sep)
                if len(parts) >= 3:
                    base_name = f"{parts[-3]}_{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
                else:
                    base_name = os.path.splitext(os.path.basename(path))[0]
                
                # Noise short path (last 3 levels)
                if batch_noise_path and os.path.exists(batch_noise_path):
                     n_parts = os.path.normpath(batch_noise_path).split(os.sep)
                     if len(n_parts) >= 3:
                         noise_short = os.path.join(n_parts[-3], n_parts[-2], n_parts[-1])
                     else:
                         noise_short = os.path.basename(batch_noise_path)
                else:
                    noise_short = "None"

                full_results = {
                    'clean': res_clean,
                    'noise': res_noise,
                    'noisy': res_noisy,
                    'meta': {
                        'clean_path': path,
                        'noise_path': batch_noise_path,
                        'noise_short_path': noise_short,
                        'base_name': base_name
                    }
                }
                
                # 5. Visualize
                # Add subdirectory based on Noise ID
                if batch_noise_path and batch_noise_path != "No_Noise_Found":
                    noise_id = os.path.splitext(os.path.basename(batch_noise_path))[0]
                    sample_save_dir = os.path.join(save_dir, noise_id)
                else:
                    sample_save_dir = save_dir
                
                os.makedirs(sample_save_dir, exist_ok=True)
                
                save_path = os.path.join(sample_save_dir, f"{base_name}_comparison.png")
                self.visualize_ig_comparison(full_results, save_path)
                print(f"[Attribution] Saved comparison: {save_path}")
                
            except Exception as e:
                print(f"[Attribution] Error analyzing {path}: {str(e)}")
                import traceback
                traceback.print_exc()

    def cleanup(self):
        pass