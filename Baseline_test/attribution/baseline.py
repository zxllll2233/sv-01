import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm


class BaselineComputer:
    """
    预计算 FBank 均值基线，替代 zero baseline。
    支持: global_mean, speaker_mean, cross_speaker_mean
    """

    def __init__(self, model, target_length=32240, device='cuda', cache_dir=None):
        """
        Args:
            model: ECAPA_TDNN 模型实例（用于其 torchfbank）
            target_length: 音频目标采样点数
            device: 计算设备
            cache_dir: 基线缓存目录（避免重复计算）
        """
        self.model = model
        self.target_length = target_length
        self.device = device
        self.cache_dir = cache_dir

        # 缓存：speaker_id -> mean_fbank_tensor [80, 1]
        self._speaker_means = {}
        # 全局均值 [80, 1]，lazy compute
        self._global_mean = None

    def _load_audio(self, audio_path):
        """加载音频并裁剪/填充到目标长度"""
        audio, sr = sf.read(audio_path)
        length = self.target_length
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        else:
            audio = audio[:length]
        return torch.FloatTensor(audio).unsqueeze(0).to(self.device)

    def _extract_fbank(self, audio_tensor):
        """提取单条音频的 FBank 特征并返回时间维均值 [80, 1]"""
        with torch.no_grad():
            fbank = self.model.torchfbank(audio_tensor) + 1e-6
            fbank = fbank.log()
            fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
            # 时间维均值: [1, 80, T] -> [80, 1]
            mean = fbank.mean(dim=2, keepdim=True).squeeze(0)
        return mean

    def compute_speaker_mean(self, speaker_audio_paths):
        """
        计算单个说话人的 FBank 均值基线。
        Args:
            speaker_audio_paths: 该说话人的音频文件路径列表
        Returns:
            mean_fbank: [80, 1] tensor
        """
        means = []
        for path in tqdm(speaker_audio_paths, desc="Computing speaker mean baseline"):
            try:
                audio = self._load_audio(path)
                mean = self._extract_fbank(audio)
                means.append(mean)
            except Exception as e:
                print(f"Warning: failed to process {path}: {e}")
                continue

        if not means:
            return torch.zeros(80, 1, device=self.device)

        speaker_mean = torch.stack(means, dim=0).mean(dim=0)
        return speaker_mean

    def compute_global_mean(self, audio_paths, max_samples=500):
        """
        计算全局 FBank 均值基线。
        Args:
            audio_paths: 音频文件路径列表（将从其中采样 max_samples 条）
            max_samples: 最大采样数
        Returns:
            mean_fbank: [80, 1] tensor
        """
        if self._global_mean is not None:
            return self._global_mean

        import random
        if len(audio_paths) > max_samples:
            audio_paths = random.sample(audio_paths, max_samples)

        means = []
        for path in tqdm(audio_paths, desc="Computing global mean baseline"):
            try:
                audio = self._load_audio(path)
                mean = self._extract_fbank(audio)
                means.append(mean)
            except Exception as e:
                print(f"Warning: failed to process {path}: {e}")
                continue

        if not means:
            return torch.zeros(80, 1, device=self.device)

        self._global_mean = torch.stack(means, dim=0).mean(dim=0)
        return self._global_mean

    def compute_cross_speaker_mean(self, audio_paths, exclude_speaker_paths=None, max_samples=200):
        """
        计算跨说话人 FBank 均值基线。
        Args:
            audio_paths: 全部音频路径
            exclude_speaker_paths: 要排除的说话人音频路径（确保不是同一说话人）
            max_samples: 最大采样数
        Returns:
            mean_fbank: [80, 1] tensor
        """
        exclude_set = set(exclude_speaker_paths) if exclude_speaker_paths else set()
        candidate_paths = [p for p in audio_paths if p not in exclude_set]

        return self.compute_global_mean(candidate_paths, max_samples=max_samples)

    def get_baseline(self, baseline_type, input_fbank_shape,
                     speaker_audio_paths=None, all_audio_paths=None,
                     exclude_speaker_paths=None):
        """
        统一接口：根据类型返回基线张量。
        Args:
            baseline_type: 'zero' | 'global_mean' | 'speaker_mean' | 'cross_speaker_mean'
            input_fbank_shape: 目标 FBank 形状 [1, 80, T]，用于广播
            speaker_audio_paths: 说话人音频路径（speaker_mean 时需要）
            all_audio_paths: 全部音频路径（global_mean / cross_speaker_mean 时需要）
            exclude_speaker_paths: 排除的说话人路径（cross_speaker_mean 时需要）
        Returns:
            baseline: [1, 80, T] tensor
        """
        T = input_fbank_shape[-1]

        if baseline_type == 'zero':
            return torch.zeros(input_fbank_shape, device=self.device)

        elif baseline_type == 'global_mean':
            mean = self.compute_global_mean(all_audio_paths)
            return mean.unsqueeze(0).expand(-1, -1, T)

        elif baseline_type == 'speaker_mean':
            mean = self.compute_speaker_mean(speaker_audio_paths)
            return mean.unsqueeze(0).expand(-1, -1, T)

        elif baseline_type == 'cross_speaker_mean':
            mean = self.compute_cross_speaker_mean(
                all_audio_paths, exclude_speaker_paths)
            return mean.unsqueeze(0).expand(-1, -1, T)

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
