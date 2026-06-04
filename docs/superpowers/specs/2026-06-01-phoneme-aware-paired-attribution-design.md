# Phoneme-Aware Paired Attribution for Interpretable Speaker Verification

**Design Spec** · 2026-06-01 · ICASSP/Interspeech submission

---

## 0. TL;DR（一页摘要）

将现有 **paired attribution** 工作升级为 **phoneme-aware** 版本，并加入 **scaled cosine objective (s=30)** 解决数值放大问题，配以司法语音学风格的可视化。最终在 ICASSP 4 页篇幅内讲一个"深度 SV 模型 × forensic phonetics 桥接"的完整故事。

**三个核心升级**：
1. **方法**：Scaled Cosine Objective + Paired Attribution（IG 公理可证）
2. **桥梁**：Phoneme-aggregated 视角，对接 forensic phonetics 数十年共识
3. **可视化**：蓝底红线 + ARPABET phoneme grid，司法实践风格

---

## 1. 论文重新定位

### 1.1 标题（候选）

**首选**：
> **Phoneme-Aware Paired Attribution for Interpretable Speaker Verification**

**备选**：
- *Bridging Deep Speaker Verification and Forensic Phonetics via Paired Attribution*
- *Where Does Voiceprint Live? Phoneme-Level Attribution for ECAPA-TDNN*

### 1.2 Abstract 草稿

> Deep speaker verification (SV) systems achieve high accuracy but remain black-box, limiting their adoption in forensic applications where the *what* and *where* of speaker-discriminative features must be explainable. Existing attribution methods (Zhang et al. 2023, PhiNet 2025) target the embedding magnitude or intermediate features, failing to localize features that are genuinely discriminative for speaker identity. We propose **Phoneme-Aware Paired Attribution**, which (1) uses a *paired difference* of Integrated Gradients on the scaled cosine similarity to isolate identity-bearing time-frequency regions, and (2) aggregates attributions by phoneme to connect deep model behavior with established forensic phonetics knowledge. On VoxCeleb1, our method achieves **X%** higher Deletion-AUC than the L2-norm baseline, and the resulting phoneme ranking shows strong agreement with forensic phonetics consensus (nasals > fricatives > vowels), providing the first quantitative bridge between black-box SV and traditional voiceprint analysis.

### 1.3 三个 Contributions（讲故事的核心）

| # | Contribution | 对应导师要求 |
|---|---|---|
| 1 | **Scaled Paired Attribution**：用 $s \cdot \cos$ 作为归因目标 + 配对差值，IG completeness 公理可证 | ③（数值放大）|
| 2 | **Phoneme-Aggregated Analysis**：首次将时频归因映射到 ARPABET 音素，与 forensic phonetics 对照 | ①（音素维度）|
| 3 | **Forensic-Style Visualization**：蓝底红线 + phoneme grid，符合司法实践范式 | ②（可视化）|

### 1.4 与已有工作的关系

| 工作 | 归因目标 | 配对 | Phoneme 视角 | Forensic-style 可视化 |
|---|---|---|---|---|
| Zhang et al. (INTERSPEECH 2023) | L2-norm | ❌ | ❌ | ❌ |
| PhiNet (arXiv 2604.01590) | 自解释中间层 | ❌ | ❌ | ❌ |
| Shen et al. (Interspeech 2025) | 多种 | ❌ | ❌ | ❌ |
| **Ours** | **scaled cos** | ✅ | ✅ | ✅ |

---

## 2. 方法设计

### 2.1 Scaled Cosine Objective（数值放大 + 方法论升级）

**当前问题**：cos 范围 [-1,1]，IG 数值稀疏（80×T ≈ 16000 bins），单 bin 归因数值 ~ 10⁻³ 量级，可视化对比度差。

**解决方案**：与 ECAPA-TDNN 训练目标对齐的 scaled cosine：

$$F(x; x_{\text{ref}}) = s \cdot \cos(\text{emb}(x), \text{emb}(x_{\text{ref}})), \quad s=30$$

`s=30` 不是拍脑袋——这是 AAM-Softmax 在 ECAPA-TDNN 训练时的默认 scale。

**为什么不破坏 IG 公理**（论文附录 0.5 页）：

**Proposition**: 设 $G(x) = s \cdot F(x)$，则 $\text{IG}^G(x) = s \cdot \text{IG}^F(x)$，且 completeness 保留：

$$\sum_i \text{IG}_i^G = s \cdot (F(x) - F(x')) = G(x) - G(x') \quad \checkmark$$

**Proof**: 由 IG 定义：
$$\text{IG}_i^G = (x_i - x'_i) \int_0^1 \frac{\partial G(x' + \alpha(x-x'))}{\partial x_i} d\alpha = s (x_i - x'_i) \int_0^1 \frac{\partial F}{\partial x_i} d\alpha = s \cdot \text{IG}_i^F$$

**效果**：
- 数值整体 ×30，95 分位 |A| 从 ~10⁻³ → ~3·10⁻²，进入可视化友好区间
- 相对 ranking 完全不变（哪些 bin 高/低排序不变）
- **与训练目标 scale 一致**，方法论闭环
- 数值层面经验证（实施后填入）：实际 max |A| 在 ±X 范围

### 2.2 Paired Attribution（核心创新，保持不变）

$$\text{Voiceprint}_i = s \cdot [\text{IG}_i^{\cos}(x, x_{\text{same}}) - \text{IG}_i^{\cos}(x, x_{\text{diff}})]$$

- 正值：同人贡献占优 → "声纹身份特征"
- 负值：异人贡献占优 → "内容/信道混淆因素"
- 0 附近：与身份无关的中性区域

### 2.3 Phoneme-Aware Aggregation（新增）

#### Step 1: Forced Alignment

工具：`facebook/wav2vec2-lv-60-espeak-cv-ft`（HuggingFace）

- 输入：raw waveform 16kHz
- 输出：每 20ms 一帧的 espeak phoneme logits
- 不需要文本转录（VoxCeleb1 无字幕，这是必选路线）

#### Step 2: Espeak → ARPABET 映射

司法语音学标准使用 ARPABET 39 音素 + sil/sp。映射表见 §6 附录。

#### Step 3: FBank 时间轴对齐

```
wav2vec2 phoneme 时间 (秒)
   ↓ × sample_rate / FBank hop (160 samples = 10ms)
FBank 帧索引 [80, T_fbank]
```

#### Step 4: Phoneme-Level Aggregation

对每个 phoneme segment $[t_s, t_e]$：

$$A_{\text{phoneme}} = \frac{1}{t_e - t_s} \sum_{t=t_s}^{t_e} \sum_{f=0}^{80} \text{Voiceprint}_{f,t}$$

每个 phoneme 类（如 /s/, /n/）聚合所有出现：

$$\bar{A}_{\text{class}} = \text{median}_{\text{instances}} A_{\text{phoneme}}^{(i)}$$

用 median 而非 mean，对 alignment 误差/异常值鲁棒。

### 2.4 Phoneme Ranking 统计

跨 500 对 trials 统计每个 ARPABET 音素的：

| 指标 | 计算 | 含义 |
|---|---|---|
| `mean_voiceprint` | $\bar{A}_{\text{class}}$ across pairs | 平均声纹贡献 |
| `consistency` | 1 - var / mean across pairs | 跨样本稳定性 |
| `top-k rank` | rank by mean_voiceprint | 论文表格用 |

输出 **ARPABET phoneme ranking table**，与 forensic phonetics 共识对比（鼻音 > 擦音 > 元音是经典 hypothesis）。

---

## 3. 可视化设计

### 3.1 司法风格 colormap 系统

**底层 FBank**：cool colormap `mpl.cm.Blues`（白→浅蓝→深蓝）

**Attribution overlay**：diverging colormap `mpl.cm.RdBu_r`（蓝→白→红），**对称归一化**：
```python
vmax = np.percentile(np.abs(A), 99)  # 99 分位裁剪，避免离群值压扁颜色
vmin = -vmax
```

**叠加策略**（mimicking 司法实践截图）：
- FBank 底图用 Blues colormap 显示能量
- Attribution 用 RdBu_r overlay，alpha = |A| / vmax（强度调透明度）
- **正归因 → 红色细条状高亮**（非块状），**负归因 → 蓝色细条**

### 3.2 主图设计（论文 Fig 2）

```
┌─────────────────────────────────────────────────────────┐
│  Trial-level (full spectrogram + attribution overlay)   │
│  ┌──────────────────────┐  ┌──────────────────────┐    │
│  │   Enroll: id10270    │  │   Test: id10270      │    │
│  │   [蓝底 + 红色条纹]   │  │   [蓝底 + 红色条纹]   │    │
│  └──────────────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Phoneme-level grid (top-5 high-attribution phonemes)   │
│  Phoneme    Enroll Seg            Test Seg              │
│  29(S)      [小图 80×t]           [小图 80×t]            │
│  23(N)      [小图 80×t]           [小图 80×t]            │
│  3(AH)      [小图 80×t]           [小图 80×t]            │
│  2(AE)      [小图 80×t]           [小图 80×t]            │
│  14(F)      [小图 80×t]           [小图 80×t]            │
└─────────────────────────────────────────────────────────┘
```

每行一个 phoneme，左右对比 enroll/test 的同音素片段。

### 3.3 Phoneme Ranking 图（论文 Fig 3）

```
┌─────────────────────────────────────────┐
│  Phoneme Ranking by Voiceprint Strength │
│                                          │
│  N  ████████████████ (nasal)            │
│  M  ███████████████  (nasal)            │
│  NG ██████████████   (nasal)            │
│  S  █████████████    (sibilant)         │
│  SH ████████████     (sibilant)         │
│  Z  ███████████      (sibilant)         │
│  R  ██████████       (liquid)           │
│  L  █████████        (liquid)           │
│  AH ████             (vowel)            │
│  IH ███              (vowel)            │
│  ...                                     │
└─────────────────────────────────────────┘
```

按 forensic phonetics 类别着色（鼻音红色、擦音橙色、元音蓝色），一图秒懂"模型学到的 vs 语音学共识"。

### 3.4 三模型对比图（论文 Fig 4）

同一个 trial 在 3 个模型上的 phoneme ranking 对比，看：
- Baseline（clean）vs Baseline (noise) vs Noise-Adversarial 模型
- 哪个模型的 phoneme attribution 更接近 forensic phonetics 共识 → 更可解释 → 更值得司法采用

### 3.5 调色板细节（CSS 化的设计 tokens）

```python
COLORS = {
    'fbank_cmap':       'Blues',       # 底层 FBank
    'attr_cmap':        'RdBu_r',      # attribution diverging
    'phoneme_nasal':    '#d62728',     # 鼻音 - 红
    'phoneme_fricative':'#ff7f0e',     # 擦音 - 橙
    'phoneme_liquid':   '#bcbd22',     # 流音 - 黄
    'phoneme_vowel':    '#1f77b4',     # 元音 - 蓝
    'phoneme_other':    '#7f7f7f',     # 其他 - 灰
}
```

---

## 4. 实验设计

### 4.1 数据

| Pool | 来源 | 数量 | 用途 |
|---|---|---|---|
| **Star samples** | 现有 `paper/pairs.csv` | 2 对 trials, 6 wav | 论文主图（Fig 2） |
| **Statistical pool** | 从 `voxceleb1_test_v2.txt` 分层抽样 | **500 对**（250 same + 250 diff），~600-800 unique wav | Phoneme ranking、Deletion/Insertion AUC |

**抽样策略**：
- 保留 pairs.csv 的 anchor speakers（id10270 系列）
- 分层覆盖 ~80-100 unique speakers
- 50/50 same/diff balance（与 EER 计算一致）

### 4.2 模型

3 个已训练的 ECAPA-TDNN：
- `Baseline_clean_noSpec_0061`（clean baseline）
- `Baseline_noise_Spec_0089`（noise + SpecAug）
- `Noise_adv_vox1_0077`（对抗训练）

### 4.3 实验 1：Phoneme Ranking（核心定量贡献）

**Hypothesis**: 我们的 paired attribution 给出的 phoneme ranking 应满足 forensic phonetics 共识：

> nasals > sibilants > liquids > vowels > stops

**测试**:
- 500 对 trials，每对计算 phoneme-aggregated attribution
- 按 ARPABET 类聚合，median + 95% CI
- 与 L2-norm baseline 对比

**预期结果表（论文 Table 1）**:

| Phoneme Class | Ours (scaled paired) | L2-norm Baseline | Forensic Consensus |
|---|---|---|---|
| Nasals (N, M, NG) | 0.XX ± 0.0X | 0.XX ± 0.0X | High |
| Sibilants (S, SH, Z) | 0.XX ± 0.0X | 0.XX ± 0.0X | High |
| Liquids (L, R) | 0.XX ± 0.0X | 0.XX ± 0.0X | Mid |
| Vowels (AH, IH, ...) | 0.XX ± 0.0X | 0.XX ± 0.0X | Low |
| Stops (P, T, K, ...) | 0.XX ± 0.0X | 0.XX ± 0.0X | Lowest |

**判定**:
- ✅ Ours 排序与 forensic consensus 一致 → 方法有效
- ✅ Ours 与 forensic 的 Spearman 相关 > L2-norm → 优于 baseline

### 4.4 实验 2：Deletion/Insertion AUC（已规划，phoneme 维度扩展）

**两个粒度**:

| 粒度 | 删除单位 | 论文意义 |
|---|---|---|
| **Bin-level**（原规划） | 单个 (freq, time) bin | 与 Zhang et al. 对齐 |
| **Phoneme-level**（新增） | 整段 phoneme | 司法实践更直观 |

**关键指标**:
- Deletion AUC: 越大越好（删除高归因区域，性能崩溃越快）
- Insertion AUC: 越大越好（恢复高归因区域，性能恢复越快）
- vs Random baseline 显著性

**预期表（论文 Table 2）**:

| Method | Bin-Del AUC | Bin-Ins AUC | Phoneme-Del AUC | Phoneme-Ins AUC |
|---|---|---|---|---|
| Random | 0.XX | 0.XX | 0.XX | 0.XX |
| L2-norm | 0.XX | 0.XX | 0.XX | 0.XX |
| **Ours (scaled paired)** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

### 4.5 实验 3：定性可视化（论文 Fig 2-4）

直接生成 §3 设计的 3 张图。

---

## 5. ICASSP 4 页结构

```
1. Introduction (0.75 页)
   - SV 可解释性现状 + Zhang/PhiNet 局限
   - Forensic phonetics 视角缺失
   - 我们的 3 个贡献

2. Method (1.25 页)
   2.1 Background: ECAPA-TDNN + IG
   2.2 Scaled Cosine Objective (s=30, IG 公理保留, Prop 1)
   2.3 Paired Attribution Δ
   2.4 Phoneme-Aware Aggregation (wav2vec2 + ARPABET)

3. Experiments (1.5 页)
   3.1 Setup (3 模型 + VoxCeleb1 + 500 对协议)
   3.2 Reliability: Deletion/Insertion AUC (Table 2)
   3.3 Phoneme Ranking (Table 1 + Fig 3)
   3.4 Qualitative: 司法风格可视化 (Fig 2, Fig 4)

4. Conclusion (0.25 页)
   - 总结 3 个贡献
   - 未来工作: speaker fingerprint matrix, cross-dataset

References (0.25 页, 12-15 引用)

附录（如有空间）:
   - Prop 1 完整证明
   - Espeak → ARPABET 映射表
```

---

## 6. 实施 Roadmap

### 阶段总览

| 阶段 | 任务 | 预计 | 产出 |
|---|---|---|---|
| **P0.1** | Phoneme alignment 脚本 + 部署到服务器 | 1 天 | `phoneme_align/` 模块 + JSON outputs |
| **P0.2** | Scaled cosine attribution（基于现有 IG） | 0.5 天 | `scaled_attribution/ig_scaled_cosine.py` |
| **P0.3** | Phoneme-level aggregation + ranking 统计 | 0.5 天 | `phoneme_aggregation/` 模块 |
| **P1.1** | 司法风格可视化模块 | 1-2 天 | `visualization/forensic_style_plot.py` + `phoneme_grid_plot.py` |
| **P1.2** | Phoneme-level Deletion/Insertion AUC | 1-2 天 | 扩展 `attribution/reliability.py` |
| **P2** | 论文写作 + 主图 finalize | 3-5 天 | ICASSP 4 页 |
| **总计** | | **8-11 工作日** | |

### P0.1 Phoneme Alignment 模块（详细）

```
scripts/phoneme_align/
├── README_server.md          # 服务器部署说明
├── env_requirements.txt      # transformers + torchaudio + phonemizer
├── espeak_to_arpabet.py      # 41 → 39 映射表
├── align_one_wav.py          # 单 wav alignment（调试用）
├── batch_align.py            # 批量对齐 + 进度条 + 断点续传
├── sample_pairs.py           # 从 voxceleb1_test_v2.txt 分层抽 500 对
└── outputs/
    ├── pairs_500.csv         # 抽样后的 500 对
    └── alignments/
        └── id10270/
            └── x6uYqmx31kE/
                └── 00001.json
```

**JSON 格式**:
```json
{
  "wav_path": "id10270/x6uYqmx31kE/00001.wav",
  "duration_sec": 8.32,
  "sample_rate": 16000,
  "fbank_hop_ms": 10,
  "phonemes": [
    {
      "arpabet": "S",
      "arpabet_id": 29,
      "espeak_raw": "s",
      "start_sec": 0.12,
      "end_sec": 0.18,
      "fbank_start_idx": 12,
      "fbank_end_idx": 18,
      "confidence": 0.92
    }
  ]
}
```

**服务器命令**（你部署后跑）:
```bash
# 1. 抽样
python scripts/phoneme_align/sample_pairs.py \
  --eval_list /home/database/sre/voxceleb/voxceleb1/voxceleb1_test_v2.txt \
  --n_pairs 500 \
  --output scripts/phoneme_align/outputs/pairs_500.csv \
  --seed 42

# 2. 批量对齐（GPU）
python scripts/phoneme_align/batch_align.py \
  --pairs_csv scripts/phoneme_align/outputs/pairs_500.csv \
  --eval_path /home/database/sre/voxceleb/voxceleb1/test/wav \
  --output_dir scripts/phoneme_align/outputs/alignments \
  --device cuda \
  --batch_size 8
```

### P0.2 Scaled Cosine Attribution

在你现有 `Baseline_test/attribution/integrated_gradients.py` 基础上加 `scale` 参数：

```python
class IntegratedGradients_ECAPA:
    def __init__(self, model, scale: float = 1.0):
        self.model = model
        self.scale = scale  # 默认 1.0 不破坏原 behavior，传 30 启用 AAM scale

    def attribute(self, x, x_ref, ...):
        # 在 cos_similarity 后 * self.scale
        ...
```

**不改变现有 IG 实现**，只加一个乘数 + completeness 校验脚本（论文附录素材）。

### P0.3 Phoneme Aggregation

输入：(attribution map 80×T) + (phoneme JSON)
输出：per-phoneme attribution scalar + per-class statistics

```python
def aggregate_by_phoneme(
    attribution: np.ndarray,    # [80, T_fbank]
    alignment: dict,             # JSON loaded
) -> dict[str, float]:
    """Return per-phoneme-instance mean attribution."""
```

### P1.1 司法风格可视化

`visualization/forensic_style_plot.py`：
- `plot_trial_overlay(fbank, attribution, save_path)` — 主图
- `plot_phoneme_grid(fbank_enroll, attr_enroll, align_enroll, fbank_test, attr_test, align_test, top_k=5, save_path)` — phoneme 对比图
- `plot_phoneme_ranking(ranking_dict, save_path)` — Fig 3

### P1.2 Phoneme-Level Reliability

扩展 `Baseline_test/attribution/reliability.py`：

```python
def deletion_test_phoneme_level(
    model, fbank, attribution, alignment,
    ratios=[0.05, 0.10, 0.20, 0.30, 0.50],
):
    """Delete top-K% phonemes (by attribution), measure performance drop."""
```

---

## 7. 风险与备选

| 风险 | 缓解策略 |
|---|---|
| **wav2vec2 phoneme 对齐精度不够（边界 > 50ms 误差）** | (1) 用 median 而非 mean 聚合，对边界误差鲁棒；(2) 必要时手工验证 20-30 个样本的对齐 |
| **500 对统计仍不够显著** | (1) 优先用 paired t-test 比较 Ours vs L2-norm 而非绝对 ranking；(2) 备选扩到 1000 对 |
| **Reviewer 质疑 s=30 选择** | (1) 附录 Prop 1 证明 + completeness 数值验证；(2) 对 s ∈ {1, 10, 30, 100} 做 sensitivity ablation |
| **Phoneme ranking 与 forensic consensus 不符** | (1) 这本身也是 finding：deep model 学到的与传统不同；(2) 论文 framing 改为 "我们发现 deep SV 模型在 X 类音素上的依赖与传统知识有偏离" |
| **Espeak → ARPABET 映射有误** | (1) 用公开映射表（如 G2P 工具的标准表）；(2) 跨 5-10 个样本人工检查 |
| **服务器 GPU 排队** | (1) wav2vec2 对齐可以在本地 Mac MPS 跑（500 对 ~30 分钟）；(2) Colab T4 备选 |

---

## 8. 不做的事（YAGNI 清单）

| 不做 | 原因 |
|---|---|
| Phoneme-conditional speaker fingerprint matrix | 体量大，留给期刊版/毕业论文 |
| 跨数据集（LibriSpeech）一致性验证 | 4 页放不下 |
| 跨架构（ResNet, WavLM-SV）对比 | 同上，且模型训练成本高 |
| 归因加权训练（Phase 2 原规划） | 4 页放不下，留作 future work |
| 全 VoxCeleb1 test set（37720 对）对齐 | 500 对统计足够，不必要 |
| 多种 baseline 对比（Expected Gradients, SmoothGrad） | 论文焦点应在 phoneme + paired，不在 IG 变体 |

---

## 9. 验收标准

本设计被认为成功执行的标准：

- [ ] Phoneme alignment 输出覆盖 500 对 trials 涉及的所有 wav
- [ ] Scaled cosine 数值验证：max |A| 在 ±0.5 ~ ±5 范围（论文友好区）
- [ ] Phoneme ranking 表（Table 1）跑出且 Ours vs L2-norm 有显著差异（p < 0.05）
- [ ] Deletion/Insertion AUC（Table 2）Ours 优于 L2-norm 和 Random 基线
- [ ] 主图（Fig 2）司法风格符合截图范式：蓝底红线 + phoneme grid
- [ ] Phoneme ranking 图（Fig 3）支持 "Ours 与 forensic consensus 一致" 或 "我们发现的偏离"
- [ ] 4 页 ICASSP draft 完成，所有图表 ready

---

## 10. 附录

### 10.1 ARPABET 39 音素 + 类别（用于排序图着色）

```
Nasals      (3): M, N, NG
Sibilants   (4): S, SH, Z, ZH
Fricatives  (4): F, V, TH, DH
Affricates  (2): CH, JH
Liquids     (2): L, R
Glides      (2): W, Y
Aspirate    (1): HH
Stops       (6): P, B, T, D, K, G
Vowels     (15): AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW
Silence     (1): SIL/SP
─────────────────────────────────
Total: 40 (39 phonemes + silence)
```

### 10.2 Espeak → ARPABET 映射（核心 30 个，完整表在脚本里）

```python
ESPEAK_TO_ARPABET = {
    # Vowels
    'a': 'AA', 'æ': 'AE', 'ʌ': 'AH', 'ə': 'AH',
    'ɔ': 'AO', 'ɛ': 'EH', 'ɪ': 'IH', 'i': 'IY',
    'oʊ': 'OW', 'ʊ': 'UH', 'u': 'UW',
    # Consonants
    'b': 'B', 'tʃ': 'CH', 'd': 'D', 'ð': 'DH',
    'f': 'F', 'g': 'G', 'h': 'HH', 'dʒ': 'JH',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N',
    'ŋ': 'NG', 'p': 'P', 'r': 'R', 's': 'S',
    'ʃ': 'SH', 't': 'T', 'θ': 'TH', 'v': 'V',
    'w': 'W', 'j': 'Y', 'z': 'Z', 'ʒ': 'ZH',
    # Silence
    '_': 'SIL', '': 'SIL',
}
```

### 10.3 服务器路径汇总

| 资源 | 路径 |
|---|---|
| VoxCeleb1 test wav | `/home/database/sre/voxceleb/voxceleb1/test/wav` |
| VoxCeleb1 dev wav | `/home/database/sre/voxceleb/voxceleb1/dev/wav` |
| test trials | `/home/database/sre/voxceleb/voxceleb1/voxceleb1_test_v2.txt` |
| MUSAN noise | `/home/database/noise/musan` |
| Model 1 (Baseline clean) | `/home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_clean_noSpec/exp/vox1/model/model_0061.model` |
| Model 2 (Baseline noise) | `/home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_noise_noSpec/vox1/clean/model/model_0078.model` |
| Model 3 (Noise adv) | `/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/3.05/model/model_0077.model` |
| 工作根目录 | `/home/zhangxl24/SpeakerRecongnition/` |
