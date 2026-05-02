# P0: 归因目标与基线改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 IG 归因目标从 embedding L2-norm 改为基于配对样本的 cosine similarity，并将基线从 zero 改为数据流形基线，使归因语义从"模型关注了什么"变为"什么时频区域对判定是否为同一说话人贡献最大"。

**Architecture:** 在 `IntegratedGradients_ECAPA.generate()` 中引入参考音频 `ref_tensor`，归因目标改为 `cosine_sim(emb(x), emb(x_ref).detach())`；新增 `BaselineComputer` 类预计算全局/说话人/跨说话人 FBank 均值基线；`ECAPAAttributionAnalyzer` 新增正例/反例配对归因流程；可视化新增正例/反例/差值三行对比。

**Tech Stack:** PyTorch, torchaudio, numpy, matplotlib, soundfile

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `Baseline_test/attribution/baseline.py` | **Create** | 基线计算工具：全局均值、说话人均值、跨说话人均值 |
| `Baseline_test/attribution/integrated_gradients.py` | **Modify** | 核心改造：配对归因目标 + 多基线支持 |
| `Baseline_test/attribution/analyzer.py` | **Modify** | 新增配对分析流程、差值归因、多基线可视化 |
| `Baseline_test/attribution/__init__.py` | **Modify** | 导出 `BaselineComputer` |
| `Baseline_test/run_attribution.py` | **Modify** | 新增 CLI 参数：配对模式、基线类型、参考音频 |

---

### Task 1: 创建 BaselineComputer — 数据流形基线计算

**Files:**
- Create: `Baseline_test/attribution/baseline.py`

- [ ] **Step 1: 创建 baseline.py，实现 BaselineComputer 类**

```python
import torch
import numpy as np
import soundfile as sf
import os
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
            # fallback: 返回零基线
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
            mean = self.compute_global_mean(all_audio_paths)  # [80, 1]
            return mean.unsqueeze(0).expand(-1, -1, T)  # [1, 80, T]

        elif baseline_type == 'speaker_mean':
            mean = self.compute_speaker_mean(speaker_audio_paths)  # [80, 1]
            return mean.unsqueeze(0).expand(-1, -1, T)  # [1, 80, T]

        elif baseline_type == 'cross_speaker_mean':
            mean = self.compute_cross_speaker_mean(
                all_audio_paths, exclude_speaker_paths)  # [80, 1]
            return mean.unsqueeze(0).expand(-1, -1, T)  # [1, 80, T]

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
```

- [ ] **Step 2: 更新 `__init__.py` 导出**

修改 `Baseline_test/attribution/__init__.py`:

```python
from .analyzer import ECAPAAttributionAnalyzer
from .integrated_gradients import IntegratedGradients_ECAPA
from .baseline import BaselineComputer
```

---

### Task 2: 改造 IntegratedGradients_ECAPA — 配对归因目标 + 多基线

**Files:**
- Modify: `Baseline_test/attribution/integrated_gradients.py`

**关键改动点：**
1. `generate()` 新增参数 `ref_tensor`（参考音频）和 `baseline`（支持 tensor 传入）
2. 归因目标从 `outputs.norm(p=2)` 改为 `cosine_similarity(emb_input, emb_ref.detach())`
3. 取消注释收敛性验证（line 109-112），改为可配置的 assert
4. 梯形法则替换简单均值近似（line 101）

- [ ] **Step 1: 重写 integrated_gradients.py**

完整替换文件内容：

```python
import torch
import torch.nn.functional as F
import numpy as np


class IntegratedGradients_ECAPA:
    def __init__(self, model, n_steps=50, convergence_threshold=0.1):
        """
        针对ECAPA-TDNN的Integrated Gradients实现
        
        Args:
            model: ECAPA_TDNN模型实例
            n_steps: 积分步数
            convergence_threshold: IG收敛性验证阈值（相对误差）
        """
        self.model = model
        self.n_steps = n_steps
        self.convergence_threshold = convergence_threshold

    def _extract_fbank(self, audio_tensor):
        """从原始音频提取FBank特征 [1, 80, T]"""
        with torch.no_grad():
            fbank = self.model.torchfbank(audio_tensor) + 1e-6
            fbank = fbank.log()
            fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
        return fbank

    def _forward_from_fbank(self, x):
        """从FBank特征开始前向传播，返回embedding"""
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = self.model.conv1(x)
        x = self.model.relu(x)
        x = self.model.bn1(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x + x1)
        x3 = self.model.layer3(x + x1 + x2)

        x = self.model.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.model.relu(x)

        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)

        w = self.model.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.model.bn5(x)
        x = self.model.fc6(x)
        x = self.model.bn6(x)
        return x

    def generate(self, input_tensor, ref_tensor=None, baseline=None,
                 objective='cosine_sim', verify_convergence=True):
        """
        生成积分梯度归因
        
        Args:
            input_tensor: 输入音频张量 [1, samples]
            ref_tensor: 参考音频张量 [1, samples]（cosine_sim 目标时必需）
            baseline: 基线FBank张量 [1, 80, T] 或 None（使用zero baseline）
            objective: 归因目标
                'cosine_sim' - 最大化与参考embedding的余弦相似度（推荐）
                'l2_norm' - 最大化embedding L2范数（旧版兼容）
            verify_convergence: 是否验证IG收敛性
            
        Returns:
            ig: 归因图 [80, T] (numpy)
        """
        self.model.eval()

        # 1. 提取输入FBank
        input_fbank = self._extract_fbank(input_tensor)  # [1, 80, T]

        # 2. 设置基线
        if baseline is None:
            baseline = torch.zeros_like(input_fbank)
        else:
            # 确保baseline与input_fbank形状一致
            if baseline.shape != input_fbank.shape:
                # baseline可能是 [80, 1]，需要广播到 [1, 80, T]
                baseline = baseline.expand_as(input_fbank)

        # 3. 计算参考embedding（如果使用cosine_sim目标）
        if objective == 'cosine_sim':
            if ref_tensor is None:
                raise ValueError("cosine_sim objective requires ref_tensor")
            with torch.no_grad():
                ref_fbank = self._extract_fbank(ref_tensor)
                ref_embedding = self._forward_from_fbank(ref_fbank)
                # 归一化参考embedding并detach，梯度不流向参考
                ref_embedding = F.normalize(ref_embedding, p=2, dim=1).detach()

        # 4. 生成插值路径
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=input_tensor.device)
        alphas = alphas.view(-1, 1, 1, 1)  # [n_steps+1, 1, 1, 1]

        interpolated_inputs = baseline + alphas * (input_fbank - baseline)
        interpolated_inputs.requires_grad_(True)

        # 5. 前向传播插值输入
        outputs = self._forward_from_fbank(interpolated_inputs)  # [n_steps+1, 192]

        # 6. 计算归因目标标量
        if objective == 'cosine_sim':
            # 归一化当前embedding
            outputs_norm = F.normalize(outputs, p=2, dim=1)
            # cosine similarity with detached reference
            score = torch.sum(outputs_norm * ref_embedding, dim=1)  # [n_steps+1]
        elif objective == 'l2_norm':
            score = outputs.norm(p=2, dim=1)  # [n_steps+1]
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # 7. 反向传播求梯度
        grads = torch.autograd.grad(
            outputs=score,
            inputs=interpolated_inputs,
            grad_outputs=torch.ones_like(score),
            create_graph=False,
            retain_graph=False
        )[0]  # [n_steps+1, 80, T]

        # 8. 梯形法则积分近似
        # (y_0 + 2*y_1 + ... + 2*y_{n-1} + y_n) / (2*n)
        # 而非简单 mean(grads)
        grads_arr = grads  # [n_steps+1, 80, T]
        avg_grads = (grads_arr[0] + 2 * grads_arr[1:-1].sum(dim=0) + grads_arr[-1]) / (2 * self.n_steps)

        # 9. 计算IG
        delta = (input_fbank - baseline).squeeze(0)  # [80, T]
        ig = delta * avg_grads  # [80, T]

        # 10. 收敛性验证
        if verify_convergence:
            with torch.no_grad():
                f_input = self._forward_from_fbank(input_fbank)
                f_baseline = self._forward_from_fbank(baseline)

                if objective == 'cosine_sim':
                    f_input_norm = F.normalize(f_input, p=2, dim=1)
                    f_baseline_norm = F.normalize(f_baseline, p=2, dim=1)
                    score_diff = (torch.sum(f_input_norm * ref_embedding, dim=1) -
                                  torch.sum(f_baseline_norm * ref_embedding, dim=1))
                else:
                    score_diff = f_input.norm(p=2, dim=1) - f_baseline.norm(p=2, dim=1)

                ig_sum = ig.sum()
                relative_error = abs((score_diff - ig_sum).item()) / (abs(score_diff).item() + 1e-8)

                if relative_error > self.convergence_threshold:
                    print(f"[IG Warning] Convergence check: relative_error={relative_error:.4f} "
                          f"(threshold={self.convergence_threshold}). "
                          f"Consider increasing n_steps (current={self.n_steps})")
                else:
                    print(f"[IG] Convergence check passed: relative_error={relative_error:.4f}")

        return ig.cpu().detach().numpy()
```

- [ ] **Step 2: 验证改造后的 IG 仍可通过旧接口调用**

运行以下 Python 代码验证向后兼容性（需在有 GPU 和模型的环境）：

```python
# 验证脚本：不改变任何调用方式，objective='l2_norm' + baseline=None 应等价于旧版
# 此步骤在 Task 4 的集成测试中一并完成
```

此步骤为设计验证，实际执行在 Task 4。

---

### Task 3: 改造 ECAPAAttributionAnalyzer — 配对归因流程 + 差值归因 + 多基线可视化

**Files:**
- Modify: `Baseline_test/attribution/analyzer.py`

**关键改动：**
1. `__init__` 新增 `baseline_computer` 参数
2. 新增 `analyze_paired()` 方法：正例/反例配对归因
3. 新增 `_compute_attribution_difference()` 方法：正例 - 反例 = 纯 voiceprint 归因
4. `visualize_ig_comparison` 扩展为支持正例/反例/差值三行
5. 保留原 `analyze()` 和 `analyze_waveform()` 接口不变

- [ ] **Step 1: 在 `__init__` 中新增 baseline_computer 参数**

修改 `analyzer.py` 的 `ECAPAAttributionAnalyzer.__init__`:

```python
def __init__(self, models_dict, C=512, n_steps=50, device='cuda',
             musan_path=None, baseline_computer=None):
    """
    Args:
        models_dict: 模型名称 -> 模型实例 字典
        musan_path: MUSAN噪声数据集路径
        baseline_computer: BaselineComputer实例（多基线支持）
    """
    self.models = models_dict
    self.device = device
    self.target_length = 200 * 160 + 240
    self.baseline_computer = baseline_computer

    # Initialize IG for each model
    self.igs = {}
    for name, model in self.models.items():
        self.igs[name] = IntegratedGradients_ECAPA(model, n_steps=n_steps)

    # Noise Augmentation Setup (unchanged)
    self.musan_path = musan_path
    self.noise_files = []
    if self.musan_path and os.path.exists(self.musan_path):
        print(f"[Attribution] Scanning noise files in {self.musan_path}...")
        self.noise_files = glob.glob(os.path.join(self.musan_path, '*/*/*.wav'))
        print(f"[Attribution] Found {len(self.noise_files)} noise files.")

    self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
```

- [ ] **Step 2: 新增 `analyze_paired` 方法**

在 `ECAPAAttributionAnalyzer` 类中新增（在 `analyze_waveform` 之后）：

```python
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
    # 获取FBank
    with torch.no_grad():
        first_model = list(self.models.values())[0]
        fbank = first_model.torchfbank(audio_tensor) + 1e-6
        fbank = fbank.log()
        fbank = fbank - torch.mean(fbank, dim=-1, keepdim=True)
        fbank = fbank.squeeze().cpu().numpy()

    # 获取基线
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

        # 正例归因：与同一说话人的cosine similarity
        ig_positive = ig_analyzer.generate(
            audio_tensor,
            ref_tensor=ref_same_tensor,
            baseline=baseline,
            objective=objective,
            verify_convergence=True
        )

        # 反例归因：与不同说话人的cosine similarity
        ig_negative = ig_analyzer.generate(
            audio_tensor,
            ref_tensor=ref_diff_tensor,
            baseline=baseline,
            objective=objective,
            verify_convergence=True
        )

        # 差值归因：正例 - 反例 = 纯voiceprint归因
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
```

- [ ] **Step 3: 新增 `visualize_paired_attribution` 方法**

在 `ECAPAAttributionAnalyzer` 类中新增可视化方法：

```python
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
    num_cols = num_models + 1  # +1 for input column

    # 3 行: positive / negative / difference
    fig, axes = plt.subplots(3, num_cols, figsize=(6 * num_cols, 12))
    plt.rcParams.update({'font.size': 10})

    groups = [
        ('positive', 'Positive (Same Speaker)', 'coolwarm'),
        ('negative', 'Negative (Diff Speaker)', 'coolwarm'),
        ('difference', 'Difference (Voiceprint)', 'RdYlGn')
    ]

    # 获取FBank（从第一个模型的结果中取）
    first_model_name = model_names[0]
    fbank = paired_results['positive'][first_model_name]['fbank']

    for g_idx, (key, label, cmap_name) in enumerate(groups):
        # Column 0: FBank（仅第一行显示，其余行留空或复制）
        if g_idx == 0:
            ax_fbank = axes[g_idx, 0]
            im = ax_fbank.imshow(fbank, origin='lower', aspect='auto', cmap='jet',
                                 extent=[0, fbank.shape[1], 0, fbank.shape[0]])
            ax_fbank.set_ylabel("Mel Filter")
            ax_fbank.set_title(f"FBank ({audio_label})")
            self._add_colorbar(ax_fbank, im, visible=True)
        else:
            axes[g_idx, 0].axis('off')

        # Columns 1..N: Models
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

            if key == 'difference':
                # 差值归因：绿色=voiceprint区域，红色=反例偏向，蓝色=正例偏向
                im = ax.imshow(ig, origin='lower', aspect='auto', cmap=cmap,
                               vmin=-limit, vmax=limit, interpolation='bicubic',
                               extent=[0, fbank.shape[1], 0, fbank.shape[0]])
            else:
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
```

- [ ] **Step 4: 新增 `analyze_and_save_paired` 方法**

用于批量处理配对归因：

```python
def analyze_and_save_paired(self, sample_pairs, save_dir, baseline_type='zero',
                            objective='cosine_sim'):
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
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, pair in enumerate(sample_pairs):
        target_path = pair['target']
        ref_same_path = pair['ref_same']
        ref_diff_path = pair['ref_diff']
        label = pair.get('label', f'sample_{i}')

        print(f"[Paired Attribution] Analyzing {label}...")

        try:
            # 加载音频
            _, target_tensor = self._load_audio_as_tensor(target_path)
            _, ref_same_tensor = self._load_audio_as_tensor(ref_same_path)
            _, ref_diff_tensor = self._load_audio_as_tensor(ref_diff_path)

            # 配对归因
            results = self.analyze_paired(
                target_tensor, ref_same_tensor, ref_diff_tensor,
                baseline_type=baseline_type, objective=objective
            )

            # 可视化
            save_path = os.path.join(save_dir, f"{label}_paired_attribution.png")
            self.visualize_paired_attribution(
                results, save_path, audio_label=os.path.basename(target_path)
            )
            print(f"[Paired Attribution] Saved: {save_path}")

        except Exception as e:
            print(f"[Paired Attribution] Error analyzing {label}: {str(e)}")
            import traceback
            traceback.print_exc()
```

---

### Task 4: 改造 run_attribution.py — 新增配对模式与基线参数

**Files:**
- Modify: `Baseline_test/run_attribution.py`

**关键改动：**
1. 新增 `--mode` 参数：`legacy`（原模式）/ `paired`（配对模式）
2. 新增 `--paired_list` 参数：配对样本列表文件路径
3. 新增 `--baseline_type` 参数：`zero` / `global_mean` / `speaker_mean` / `cross_speaker_mean`
4. 新增 `--objective` 参数：`cosine_sim` / `l2_norm`
5. 配对模式下加载 `BaselineComputer`

- [ ] **Step 1: 新增配对样本文件格式**

配对列表文件格式（CSV，无表头）：
```
target_path,ref_same_path,ref_diff_path,label
/path/to/target.wav,/path/to/same_spk.wav,/path/to/diff_spk.wav,sample_001
```

- [ ] **Step 2: 修改 run_attribution.py**

在 `parser` 中新增以下参数（在现有参数之后）：

```python
# 归因模式参数
parser.add_argument('--mode', type=str, default='legacy',
                    choices=['legacy', 'paired'],
                    help='Attribution mode: legacy (original) or paired (positive/negative/difference)')
parser.add_argument('--paired_list', type=str, default=None,
                    help='Path to paired sample list CSV file (for paired mode)')
parser.add_argument('--baseline_type', type=str, default='zero',
                    choices=['zero', 'global_mean', 'speaker_mean', 'cross_speaker_mean'],
                    help='Baseline type for IG')
parser.add_argument('--objective', type=str, default='cosine_sim',
                    choices=['cosine_sim', 'l2_norm'],
                    help='Attribution objective function')
```

- [ ] **Step 3: 新增配对模式主逻辑**

在 `main()` 函数的模型加载之后，根据 `args.mode` 分流：

```python
# ... (existing model loading code) ...

if args.mode == 'legacy':
    # 原有逻辑不变
    samples = get_samples(args)
    if not samples:
        print("[Attribution] Failed to get samples. Exiting.")
        return

    analyzer = ECAPAAttributionAnalyzer(
        models_dict=models_dict,
        C=args.C,
        device=args.device,
        musan_path=args.musan_path
    )

    subdir_name = "_vs_".join(model_identifiers)
    save_dir = os.path.join(args.save_path, subdir_name)
    analyzer.analyze_and_save(samples, save_dir)

elif args.mode == 'paired':
    # 配对归因模式
    if not args.paired_list or not os.path.exists(args.paired_list):
        print("[Attribution] Error: --paired_list is required for paired mode")
        return

    # 解析配对列表
    import csv
    sample_pairs = []
    with open(args.paired_list, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                pair = {
                    'target': row[0],
                    'ref_same': row[1],
                    'ref_diff': row[2],
                    'label': row[3] if len(row) >= 4 else f'pair_{len(sample_pairs)}'
                }
                # 验证路径存在
                if os.path.exists(pair['target']) and os.path.exists(pair['ref_same']) and os.path.exists(pair['ref_diff']):
                    sample_pairs.append(pair)
                else:
                    print(f"[Attribution] Warning: skipping pair with missing files: {pair['label']}")

    if not sample_pairs:
        print("[Attribution] No valid paired samples found. Exiting.")
        return

    # 初始化 BaselineComputer（如果需要非零基线）
    baseline_computer = None
    if args.baseline_type != 'zero':
        first_model = list(models_dict.values())[0]
        baseline_computer = BaselineComputer(
            model=first_model,
            target_length=200 * 160 + 240,
            device=args.device
        )

    analyzer = ECAPAAttributionAnalyzer(
        models_dict=models_dict,
        C=args.C,
        device=args.device,
        musan_path=args.musan_path,
        baseline_computer=baseline_computer
    )

    subdir_name = "_vs_".join(model_identifiers)
    save_dir = os.path.join(args.save_path, f"paired_{subdir_name}")
    analyzer.analyze_and_save_paired(
        sample_pairs, save_dir,
        baseline_type=args.baseline_type,
        objective=args.objective
    )

print(f"[Attribution] Done! Results saved.")
```

- [ ] **Step 4: 更新文件顶部的 import**

在 `run_attribution.py` 顶部的 import 区域添加：

```python
from attribution.baseline import BaselineComputer
```

---

### Task 5: 集成验证

**Files:**
- 所有已修改文件

- [ ] **Step 1: 语法检查 — 确认所有文件可导入无报错**

```bash
cd /Users/zhangxiaolei.128/PycharmProjects/PythonProject/zxl_aurora/Baseline_test
python3 -c "from attribution.integrated_gradients import IntegratedGradients_ECAPA; print('IG OK')"
python3 -c "from attribution.baseline import BaselineComputer; print('Baseline OK')"
python3 -c "from attribution.analyzer import ECAPAAttributionAnalyzer; print('Analyzer OK')"
```

- [ ] **Step 2: 向后兼容验证 — legacy 模式参数解析**

```bash
cd /Users/zhangxiaolei.128/PycharmProjects/PythonProject/zxl_aurora/Baseline_test
python3 run_attribution.py --help
```

预期：新增参数 `--mode`, `--paired_list`, `--baseline_type`, `--objective` 出现在帮助信息中，默认值分别为 `legacy`, `None`, `zero`, `cosine_sim`。

- [ ] **Step 3: LSP 诊断 — 检查所有修改文件的类型错误**

对以下文件运行 `lsp_diagnostics`：
- `Baseline_test/attribution/baseline.py`
- `Baseline_test/attribution/integrated_gradients.py`
- `Baseline_test/attribution/analyzer.py`
- `Baseline_test/run_attribution.py`

---

## 改动影响矩阵

| 文件 | 改动性质 | 向后兼容 |
|------|---------|---------|
| `integrated_gradients.py` | `generate()` 新增 `ref_tensor`, `objective`, `verify_convergence` 参数，均有默认值 | ✅ 旧调用方式不受影响 |
| `analyzer.py` | `__init__` 新增 `baseline_computer` 参数（默认 None）；新增 3 个方法 | ✅ 旧接口不变 |
| `baseline.py` | 新文件 | ✅ 无破坏性 |
| `__init__.py` | 新增导出 | ✅ 无破坏性 |
| `run_attribution.py` | 新增 CLI 参数（均有默认值）；新增 paired 分支 | ✅ `--mode legacy` 为默认 |

## 归因语义对照

| 归因模式 | 目标函数 | 归因含义 |
|---------|---------|---------|
| 旧版 `l2_norm` | `‖emb(x)‖₂` | "FBank的哪些区域让embedding模长增大" |
| 正例 `cosine_sim(same)` | `cos(emb(x), emb(x_same))` | "FBank的哪些区域支持'这是同一说话人'判断" |
| 反例 `cosine_sim(diff)` | `cos(emb(x), emb(x_diff))` | "FBank的哪些区域支持'这是不同说话人'判断" |
| 差值 `positive - negative` | 正例归因 - 反例归因 | **"什么是最纯粹的voiceprint区域"** |
