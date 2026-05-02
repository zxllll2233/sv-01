import os
import glob
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # 非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import soundfile as sf
from typing import List, Dict, Union
import time

from .integrated_gradients import IntegratedGradients_ECAPA

class ECAPAAttributionAnalyzer:
    def __init__(self, models_dict: Dict[str, torch.nn.Module], C=512, n_steps=50, device='cuda', musan_path=None, baseline_computer=None):
        """
        Args:
            models_dict: A dictionary mapping model names to model instances.
            musan_path: Path to MUSAN noise dataset for augmentation analysis.
            baseline_computer: BaselineComputer实例（多基线支持）
        """
        self.models = models_dict
        self.device = device
        self.target_length = 200 * 160 + 240 # 32240 samples
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
        """Standard loading and padding for single file"""
        audio, sr = sf.read(audio_path)
        length = self.target_length
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        else:
            # Random crop usually, but for consistent analysis let's center crop or start crop?
            # dataLoader uses random crop. For analysis, let's use start crop for determinism
            # or random? Let's stick to start crop or maybe center.
            # But wait, original code used `audio[:length]`. Let's stick to that.
            audio = audio[:length]
        
        return audio, torch.FloatTensor(audio).unsqueeze(0).to(self.device)

    def _get_noise_audio(self):
        """Randomly select a noise file and load it formatted"""
        if not self.noise_files:
            return None, None
            
        noise_path = random.choice(self.noise_files)
        noise_wav, _ = self._load_audio_as_tensor(noise_path)
        return noise_wav, noise_path

    def _augment_audio(self, clean_wav, noise_wav, noise_path):
        """
        Mix clean and noise.
        Returns: noisy_wav (numpy), noise_scaled (numpy)
        """
        # Identify noise category from path structure: .../musan/category/folder/file.wav
        # Usually split by sep.
        # dataLoader: file.split('/')[-3] -> category
        try:
            parts = noise_path.split(os.sep)
            # finding 'musan' index might be safer, but let's assume standard structure
            # If path is /.../musan/noise/free-sound/noise-001.wav
            # category is 'noise' (index -3)
            category = parts[-3]
            if category not in self.noisesnr:
                category = 'noise' # Default
        except:
            category = 'noise'
            
        clean_db = 10 * np.log10(np.mean(clean_wav ** 2) + 1e-4)
        noise_db = 10 * np.log10(np.mean(noise_wav ** 2) + 1e-4)
        
        snr_range = self.noisesnr.get(category, [0, 15])
        snr = random.uniform(snr_range[0], snr_range[1])
        
        scale = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10))
        noise_scaled = scale * noise_wav
        noisy_wav = clean_wav + noise_scaled
        
        return noisy_wav, noise_scaled, category

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
                       baseline_type='zero', objective='cosine_sim'):
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

        if baseline_type != 'zero' and self.baseline_computer is not None:
            baseline = self.baseline_computer.get_baseline(
                baseline_type, input_fbank_shape=[1, 80, fbank.shape[-1]]
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
            'baseline_type': baseline_type
        }

    def visualize_paired_attribution(self, paired_results, save_path,
                                     audio_label="Target"):
        """
        可视化配对归因结果：正例 / 反例 / 差值 三行 × (输入 + N模型) 列

        Args:
            paired_results: analyze_paired() 的返回结果
            save_path: 保存路径
            audio_label: 音频标签
        """
        model_names = list(self.models.keys())
        num_models = len(model_names)
        num_cols = num_models + 1

        fig, axes = plt.subplots(3, num_cols, figsize=(6 * num_cols, 12))
        plt.rcParams.update({'font.size': 10})

        groups = [
            ('positive', 'Positive (Same Speaker)', 'coolwarm'),
            ('negative', 'Negative (Diff Speaker)', 'coolwarm'),
            ('difference', 'Difference (Voiceprint)', 'RdYlGn')
        ]

        first_model_name = model_names[0]
        fbank = paired_results['positive'][first_model_name]['fbank']

        for g_idx, (key, label, cmap_name) in enumerate(groups):
            if g_idx == 0:
                ax_fbank = axes[g_idx, 0]
                im = ax_fbank.imshow(fbank, origin='lower', aspect='auto', cmap='jet',
                                     extent=[0, fbank.shape[1], 0, fbank.shape[0]])
                ax_fbank.set_ylabel("Mel Filter")
                ax_fbank.set_title(f"FBank ({audio_label})")
                self._add_colorbar(ax_fbank, im, visible=True)
            else:
                axes[g_idx, 0].axis('off')

            for m_idx, name in enumerate(model_names):
                col_idx = m_idx + 1
                ax = axes[g_idx, col_idx]

                if key == 'difference':
                    ig = paired_results[key][name]['ig_diff']
                else:
                    ig = paired_results[key][name]['ig']

                if ig.ndim == 3 and ig.shape[0] == 1:
                    ig = ig.squeeze(0)

                limit = max(np.percentile(np.abs(ig), 99), 1e-8)
                cmap = plt.get_cmap(cmap_name)

                im = ax.imshow(ig, origin='lower', aspect='auto', cmap=cmap,
                               vmin=-limit, vmax=limit, interpolation='bicubic',
                               extent=[0, fbank.shape[1], 0, fbank.shape[0]])

                ax.set_title(f"{name}\n{label}")
                if g_idx == 2:
                    ax.set_xlabel("Time (Frames)")
                self._add_colorbar(ax, im, visible=True)

        baseline_type = paired_results.get('baseline_type', 'zero')
        fig.suptitle(f"Paired Attribution Analysis (Baseline: {baseline_type})", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def analyze_and_save_paired(self, sample_pairs, save_dir, baseline_type='zero',
                                objective='cosine_sim', base_path=None):
        """
        批量配对归因分析

        Args:
            sample_pairs: 列表，每个元素为字典：
                {
                    'target': 'path/to/target.wav',
                    'ref_same': 'path/to/same_speaker.wav',
                    'ref_diff': 'path/to/diff_speaker.wav',
                    'label': 'optional_label'
                }
            save_dir: 保存目录
            baseline_type: 基线类型
            objective: 归因目标
            base_path: 音频路径前缀，若提供则拼接到每个相对路径前
        """
        os.makedirs(save_dir, exist_ok=True)

        for i, pair in enumerate(sample_pairs):
            target_path = os.path.join(base_path, pair['target']) if base_path else pair['target']
            ref_same_path = os.path.join(base_path, pair['ref_same']) if base_path else pair['ref_same']
            ref_diff_path = os.path.join(base_path, pair['ref_diff']) if base_path else pair['ref_diff']
            label = pair.get('label', f'sample_{i}')

            print(f"[Paired Attribution] Analyzing {label}...")

            try:
                _, target_tensor = self._load_audio_as_tensor(target_path)
                _, ref_same_tensor = self._load_audio_as_tensor(ref_same_path)
                _, ref_diff_tensor = self._load_audio_as_tensor(ref_diff_path)

                results = self.analyze_paired(
                    target_tensor, ref_same_tensor, ref_diff_tensor,
                    baseline_type=baseline_type, objective=objective
                )

                save_path = os.path.join(save_dir, f"{label}_paired_attribution.png")
                self.visualize_paired_attribution(
                    results, save_path, audio_label=os.path.basename(target_path)
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