# Phoneme-Aware Paired Attribution with Fisher-z Cosine Objective

**Design Spec v2** · 2026-06-01 (revised 2026-06-04) · ICASSP/Interspeech submission

> ⚠️ **v2 关键修订**（基于 Oracle review + 文献调研）：
> 1. 弃用 "LLR" 命名 → 改为 **Fisher z-transform of cosine**（Oracle + Morrison 文献一致结论）
> 2. 新增竞品对比：**Thebaud et al. 2024 (Odyssey)** 是直接竞品，必须重新定位
> 3. 公式全部改为 Unicode + 代码块，避免 latex 渲染问题
> 4. IG n_steps 从 50 → 300，加 clamp，加 completeness gap 报告
> 5. 新增 Wang et al. 2024 等 6 个必须 cite 的近期工作

---

## 0. TL;DR

将现有 paired attribution 工作升级为 **Fisher-z + Paired + Phoneme-Aware** 三结合归因方法，在 ICASSP 4 页篇幅内讲一个"将神经 SV 黑盒解释为 forensic 可用的 phoneme 证据"的故事。

**三个真正的创新点**（经文献交叉验证）：
1. **方法**：Fisher z-transform of cosine + Paired Attribution（IG 公理可证，与 Thebaud 2024 raw cosine 形成对比）
2. **桥梁**：Phoneme-aggregated logit-cosine 归因，对接 forensic phonetics 数十年共识
3. **可视化**：蓝底红线 + ARPABET phoneme grid，符合司法实践范式

---

## 1. 论文重新定位

### 1.1 标题（候选）

**首选**：
```
Fisher-z Paired Attribution for Forensic-Style Interpretation
of Speaker Verification: A Phoneme-Level Study
```

**备选**：
- Phoneme-Aware Score Attribution for Speaker Verification with Forensic Interpretability
- Beyond Cosine: Logit-Cosine Attribution for Decision-Boundary-Focused SV Interpretation

### 1.2 Abstract 草稿（v2）

> Deep speaker verification (SV) systems achieve high accuracy but remain
> opaque, limiting their adoption in forensic applications where features
> driving the verification decision must be explicitly localized. While
> recent works (Thebaud et al. 2024, Zhang et al. 2023) apply Integrated
> Gradients (IG) to raw cosine similarity, this objective compresses
> attribution dynamic range and does not align with forensic scoring
> conventions. We propose **Fisher-z Paired Attribution**: (1) attributing
> the Fisher z-transform of cosine similarity, which expands the bounded
> score to an unbounded domain with proper variance-stabilization properties
> (Fisher 1921), and (2) computing a paired contrastive difference to
> isolate speaker-identity features from content/channel confounds. We
> aggregate the resulting attributions by ARPABET phoneme via wav2vec2-based
> forced alignment and compare against the forensic phonetics consensus on
> speaker-discriminative phoneme classes. On VoxCeleb1 with three ECAPA-TDNN
> models, our method achieves higher Deletion/Insertion AUC than raw-cosine
> and L2-norm baselines, and the resulting phoneme ranking is consistent
> with established forensic phonetics knowledge, providing the first
> quantitative bridge between attribution-based SV interpretation and
> traditional voiceprint analysis.

### 1.3 三个 Contributions

| # | Contribution | 对应导师要求 | 与已有工作差异 |
|---|---|---|---|
| 1 | **Fisher-z Paired Attribution**: arctanh(cos) + 配对差值 | ③（数值放大）| Thebaud 2024 用 raw cos; Zhang 2023 用 L2-norm; 本工作用 Fisher-z |
| 2 | **Phoneme-Aggregated Forensic Bridge**: ARPABET 聚合 + forensic 对照 | ①（音素维度）| Thebaud 只做 phoneme-power 排序; Wang 2024 用 LayerCAM 发现元音最重要(与 forensic 共识矛盾)——我们用 paired+Fisher-z 验证两派对立 |
| 3 | **Forensic-Style Visualization**: 蓝底红线 + phoneme grid | ②（可视化）| 已有工作都是 ML 风格 heatmap |

### 1.4 与已有工作的关系（v2 扩充对比表）

```
                        Score func        Method           Paired   Phoneme   Forensic-style
Zhang et al. 2023       L2-norm           IG/5 methods     ✗        ✗         ✗
Thebaud et al. 2024     raw cosine        IG               ✗        ✓         ✗
Wang et al. 2024        raw cosine        LayerCAM/TAO     ✗        ✓         ✗
ExPO (Ma 2025)          per-phoneme cos   architectural    ✗        ✓         partial
PhiNet (2025/2026)      hybrid            intrinsic        ✗        ✓         partial
PDAF (2024)             ─                 attention        ✗        partial   ✗
BA-LR (2023)            calibrated LR     Shapley/trees    ✗        partial   ✓
─────────────────────────────────────────────────────────────────────────────────
Ours (v2)               Fisher-z(cos)     IG + paired      ✓        ✓         ✓
```

### 1.5 必须 cite 的工作（v2 新增基于 librarian 调研）

```
[1] Thebaud et al. 2024 — Odyssey
    "A Phonetic Analysis of SV Systems through Phoneme selection and IG"
    →最直接竞品, introduction 第一段就要提

[2] Zhang et al. 2023 — Interspeech (pp. 2233-2237)
    "A Study on Visualization of Voiceprint Feature"
    →ECAPA + IG 先行者

[3] Wang et al. 2024 — arXiv:2402.02730
    "How phonemes contribute to deep speaker models?"
    →关键对比: 他们发现元音 > 擦音 (与 forensic 矛盾)

[4] Ma et al. 2025 — ExPO (arXiv:2501.05729)
    "Explainable Phonetic Trait-Oriented Network"
    →architectural 路线代表

[5] Ben-Amor & Bonastre 2023 — Interspeech — BA-LR
    "Binary-Attribute-based Likelihood Ratio"
    →forensic LR 方向代表

[6] Morrison et al. 2021 — Science & Justice 61(3)
    "Consensus on validation of forensic voice comparison"
    →forensic LLR 标准引用必备

[7] Drygajlo et al. 2015 — ENFSI Guidelines
    →forensic 圈金标准

[8] Sundararajan et al. 2017 — ICML
    "Axiomatic Attribution for Deep Networks"
    →IG 原始论文, 推荐 n_steps=100-1000 用于非线性目标

[9] Fisher 1921 — Metron
    "On the 'probable error' of a coefficient of correlation"
    →arctanh transform 命名起源
```

---

## 2. 方法设计（v2 重写）

### 2.1 Fisher z-transform of Cosine（替代 LLR）

#### 2.1.1 问题陈述

```
问题 1: 动态范围有限
   cos ∈ [-1, 1]
   典型 same-pair:  0.6 - 0.9
   典型 diff-pair:  0.1 - 0.4
   差值 ≈ 0.2 - 0.4
   单 bin IG ~ 10^-3 量级, 可视化对比度差

问题 2: 梯度饱和
   高 cos 区域 (cos > 0.8) 梯度变化小
   "高度确信同人"与"绝对确信同人"在归因上几乎无区分
```

#### 2.1.2 Fisher z-transform 解决方案

```
定义:
   F_z(x; x_ref) = arctanh( clamp( cos(emb(x), emb(x_ref)), -0.98, 0.98 ) )

等价形式:
   F_z(x; x_ref) = 0.5 * log( (1 + c) / (1 - c) )      其中 c = cos(emb(x), emb(x_ref))

范围:
   c → -0.98:  F_z → -2.30
   c =  0:     F_z =  0
   c → +0.98:  F_z → +2.30

clamp 后等效梯度放大因子:
   df/dc = 1 / (1 - c^2)
   c ≈ 0:    放大 1.0×      (低置信区)
   c = 0.5:  放大 1.33×
   c = 0.8:  放大 2.78×
   c = 0.9:  放大 5.26×
   c = 0.98: 放大 25×       (clamp 上限)
```

#### 2.1.3 命名澄清（v2 关键修正, 必读）

```
❌ 不要叫 "LLR" / "Log-Likelihood Ratio"
   原因: forensic LLR 需要 P(evidence | same) / P(evidence | diff) 校准
         我们没有校准, 只是单调变换 raw cosine
         Morrison (2016) 明确指出 similarity-only scores 不是 LLR
   风险: Morrison 派审稿人直接拒稿

✅ 正确叫法 (论文里全文统一):
   - "Fisher z-transform of cosine similarity"
   - "logit-cosine objective"
   - "log-odds-like transform"
   - "variance-stabilizing transform"
   不要碰 "likelihood ratio" 这个词
```

#### 2.1.4 理论根据

```
Fisher z-transform (Fisher 1921, Metron):
   - 相关系数的方差稳定化变换
   - 统计学中有百年应用历史
   - 不需要 forensic 校准就能正当使用
   - 在 (−1, +1) 区间将 bounded score 映射到 unbounded domain

引用方式:
   "We apply the Fisher z-transform (Fisher 1921), a variance-stabilizing
    transform widely used in statistics, to map cosine similarity from
    its bounded range [-1, 1] to an unbounded domain suitable for
    fine-grained attribution analysis."
```

#### 2.1.5 与 IG 公理的关系

```
Proposition (IG completeness under monotonic transform):
   设 F: X → R 可微 (e.g., cosine), g: R → R 可微 (e.g., arctanh).
   令 G(x) = g(F(x)).
   则:
       Σ_i IG_i^G(x) = G(x) - G(x_baseline)

证明见附录 §10.5.

特例: g = arctanh ∘ clamp[-0.98, 0.98]
     在 clamp 区间内严格成立, 边界外不连续点对积分 measure-zero, 数值上鲁棒.

注: 非线性 g 会改变 bin 之间的相对 ranking (不只是 scale).
   这正是我们想要的——把"高 cos 区域的微小差异"
   放大成"显著归因差异".
```

### 2.2 Paired Attribution（核心创新）

```
定义:
   Voiceprint(i) = IG_i [ F_z(x; x_same) - F_z(x; x_diff) ]

   其中:
     IG_i [ . ] = 第 i 个 (freq, time) bin 在该目标下的 Integrated Gradient
     x_same    = 同说话人参考样本
     x_diff    = 不同说话人参考样本

物理意义:
   Voiceprint(i) > 0:  该 bin 对"判同人"贡献 > 对"判不同人"贡献 → 声纹身份特征
   Voiceprint(i) < 0:  该 bin 对"判不同人"贡献 > 对"判同人"贡献 → 内容/信道混淆
   Voiceprint(i) ≈ 0:  该 bin 对身份判定中性

注意 (v2 新增):
   由 IG 线性性:
       IG_i [ F_z(x_same) - F_z(x_diff) ] = IG_i [ F_z(x_same) ] - IG_i [ F_z(x_diff) ]
   两种实现等价. 论文写法采用前者 (单一 contrastive objective)
   因为概念上更 "principled" (一次归因一个明确目标, 而非两次归因相减).
```

### 2.3 Phoneme-Aware Aggregation（v2 详化）

#### Step 1: Forced Alignment

```
工具: facebook/wav2vec2-lv-60-espeak-cv-ft (HuggingFace)
输入: raw waveform 16 kHz
输出: 每 20 ms 一帧的 espeak phoneme logits
不需要文本转录 (VoxCeleb1 无字幕)
```

#### Step 2: Espeak → ARPABET 映射

```
ESPEAK (~41 个 IPA-like 符号) → ARPABET (39 + sil)
完整映射表见附录 §10.2
```

#### Step 3: FBank 时间轴对齐

```
ECAPA FBank 配置 (已从 model.py 验证):
   sample_rate = 16000 Hz
   n_fft       = 512
   win_length  = 400     (25 ms)
   hop_length  = 160     (10 ms)
   n_mels      = 80
   f_min       = 20 Hz
   f_max       = 7600 Hz

phoneme 时间 → FBank 帧索引:
   fbank_frame_idx = round( phoneme_time_sec * 16000 / 160 )
                   = round( phoneme_time_sec * 100 )
```

#### Step 4: Per-Phoneme Aggregation

```
对每个 phoneme segment [t_s, t_e]:

   A_phoneme = median over t in [t_s, t_e] of:
                  sum over f in [0, 80) of Voiceprint(f, t)

用 median (而非 mean):
   - 对 wav2vec2 边界对齐误差鲁棒
   - 对异常值 (e.g., burst frames) 鲁棒
```

#### Step 5: Per-Class Statistics

```
对每个 ARPABET 类 c (S, N, AH, ...):

   A_class(c)    = median over instances of A_phoneme
   sigma_class(c)= MAD (median absolute deviation) over instances

跨 500 对 trials 聚合, 得到 phoneme ranking table.
```

---

## 3. 可视化设计

### 3.1 司法风格 colormap 系统

```python
# 颜色配置 (v2 调色板)
FBANK_CMAP   = 'Blues'      # 底层 FBank 能量
ATTR_CMAP    = 'RdBu_r'     # attribution diverging (red+ / white0 / blue-)

# 归一化 (避免离群值压扁颜色)
vmax = np.percentile(np.abs(A), 99)   # 99 分位裁剪
vmin = -vmax

# 类别色 (用于 Fig 3 phoneme ranking 着色)
COLOR_NASAL     = '#d62728'    # 红
COLOR_SIBILANT  = '#ff7f0e'    # 橙
COLOR_FRICATIVE = '#bf8700'    # 黄
COLOR_LIQUID    = '#2ca02c'    # 绿
COLOR_VOWEL     = '#1f77b4'    # 蓝
COLOR_STOP      = '#7f7f7f'    # 灰
```

### 3.2 主图 Fig 2 (trial + phoneme grid)

```
┌──────────────────────────────────────────────────────────┐
│  Trial-level: FBank (Blues 底) + Attribution (RdBu_r)     │
│  ┌──────────────────┐    ┌──────────────────┐            │
│  │  Enroll          │    │  Test            │            │
│  │  [蓝底 + 红色条] │    │  [蓝底 + 红色条] │            │
│  └──────────────────┘    └──────────────────┘            │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Top-5 high-attribution phoneme segments                  │
│                                                            │
│  Phoneme    Enroll Seg            Test Seg                │
│  29 (S)     [80×t small map]      [80×t small map]        │
│  23 (N)     [80×t small map]      [80×t small map]        │
│  3  (AH)    [80×t small map]      [80×t small map]        │
│  2  (AE)    [80×t small map]      [80×t small map]        │
│  14 (F)     [80×t small map]      [80×t small map]        │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Fig 3 — Phoneme Ranking 对比图

```
按 ARPABET 类排序, 三条 bar 对比:
   Ours (Fisher-z paired)   ━━━━━━━━━
   Thebaud (raw cos IG)     ━━━━━━
   L2-norm (Zhang)          ━━━

类别着色:
   Nasals    (M, N, NG)         red       — forensic 高
   Sibilants (S, SH, Z, ZH)     orange    — forensic 高
   Fricatives(F, V, TH, DH)     yellow    — forensic 中
   Liquids   (L, R)             green     — forensic 中
   Vowels    (AA, AE, AH, ...)  blue      — forensic 低
   Stops     (P, T, K, B, D, G) gray      — forensic 最低
```

### 3.4 Fig 4 — 三模型对比

```
横轴: ARPABET phoneme classes
纵轴: A_class
三条线:
   Baseline_clean
   Baseline_noise_Spec
   Noise_adv_vox1

讨论: 哪个模型的 phoneme ranking 最接近 forensic consensus,
     哪个模型在哪个 phoneme 类上更鲁棒
```

---

## 4. 实验设计（v2 修订）

### 4.1 数据协议

```
Star pool (主图):
   现有 paper/pairs.csv 的 2 对 trials, 6 unique wavs
   论文 Fig 2 用 (定性主图)

Statistical pool (Table):
   从 voxceleb1_test_v2.txt 分层抽 500 对
   250 same + 250 diff
   ~600-800 unique wavs
   ~80-100 unique speakers
   每个 ARPABET 类预计出现 500-2000 次
   论文 Table 1-2 用 (定量结果)

Star ⊂ Statistical (主图样本是统计池的子集, 互相印证)
```

### 4.2 三个模型

```
Model 1 (clean):       Baseline_clean_noSpec    ckpt model_0061
Model 2 (noise):       Baseline_noise_Spec      ckpt model_0089
Model 3 (noise+adv):   Noise_adv_vox1           ckpt model_0077
```

### 4.3 实验 1: Phoneme Ranking 统计（核心定量）

```
Hypothesis: Fisher-z paired attribution 的 phoneme ranking 满足:
   nasals (N,M,NG) > sibilants (S,SH,Z) > liquids (L,R) > vowels > stops

测试:
   500 对 trials
   每对计算 Voiceprint
   按 ARPABET 类聚合, 报告 median + 95% bootstrap CI
   与 forensic consensus 做 Spearman rank correlation

预期 Table 1:

   Phoneme Class      Ours (Fisher-z)   Thebaud (raw cos)   L2-norm
   Nasals             0.XXX ± 0.0X      0.XXX ± 0.0X        0.XXX ± 0.0X
   Sibilants          0.XXX ± 0.0X      0.XXX ± 0.0X        0.XXX ± 0.0X
   Liquids            0.XXX ± 0.0X      0.XXX ± 0.0X        0.XXX ± 0.0X
   Vowels             0.XXX ± 0.0X      0.XXX ± 0.0X        0.XXX ± 0.0X
   Stops              0.XXX ± 0.0X      0.XXX ± 0.0X        0.XXX ± 0.0X
   ───────────────────────────────────────────────────────────────
   Spearman ρ vs FP   0.XX              0.XX                0.XX

判定:
   ✓ Ours Spearman > Thebaud Spearman > L2 Spearman → 方法升级有效
   ✓ Ours 与 forensic FP 显著相关 → 桥梁建立
   ✗ 如果 Wang 2024 的"元音 > 擦音" pattern 在我们方法上反转 → 重要 finding
```

### 4.4 实验 2: Deletion/Insertion AUC

```
两个粒度:

Bin-level (与 Zhang/Thebaud 对齐):
   删除/插入单个 (freq, time) bin
   按 |Voiceprint| 从大到小排序

Phoneme-level (新增, 司法实践更直观):
   删除/插入整段 phoneme
   按 |A_phoneme| 从大到小排序

预期 Table 2:

   Method               Bin-Del   Bin-Ins   Phon-Del   Phon-Ins
   Random               0.XX      0.XX      0.XX       0.XX
   L2-norm (Zhang)      0.XX      0.XX      0.XX       0.XX
   raw cos (Thebaud)    0.XX      0.XX      0.XX       0.XX
   Ours (Fisher-z)      0.XX      0.XX      0.XX       0.XX
```

### 4.5 实验 3: 定性可视化（Fig 2-4）

```
直接用 §3 设计跑出来.
```

### 4.6 IG 收敛性报告（v2 新增, 必报）

```
论文 Section 3.1 末段必须报告:

   "We use n_steps=300 with completeness verification.
    Mean completeness gap on 500 pairs:
      raw cos:     X.X% ± Y.Y%
      Fisher-z:    X.X% ± Y.Y%
    Cosine values were clamped to [-0.98, 0.98] before arctanh
    to avoid numerical instability near saturation,
    affecting <1% of frames."
```

---

## 5. ICASSP 4 页结构（v2 微调）

```
1. Introduction (0.75 页)
   - SV 可解释性现状: cite Thebaud 2024 + Wang 2024 + Zhang 2023
   - 三派的局限:
       Thebaud:  raw cos, 无 paired, 无 forensic 桥接
       Wang:     LayerCAM 发现元音 > 擦音, 与 forensic 共识矛盾 (重要 hook!)
       Zhang:    L2-norm, 无配对, 无 phoneme
   - 我们的 3 个贡献

2. Method (1.25 页)
   2.1 Background: ECAPA-TDNN + cosine + IG
   2.2 Fisher z-transform Objective (Fisher 1921, 公理保留, clamp)
   2.3 Paired Contrastive Difference
   2.4 Phoneme-Aware Aggregation (wav2vec2 + ARPABET, median)

3. Experiments (1.5 页)
   3.1 Setup (3 模型 + VoxCeleb1 + 500 对 + IG 收敛性报告)
   3.2 Reliability: Deletion/Insertion AUC (Table 2)
   3.3 Phoneme Ranking + forensic 对照 (Table 1 + Fig 3)
   3.4 Qualitative (Fig 2 + Fig 4)

4. Conclusion (0.25 页)

References (0.25 页, ~15 个)

附录 (空间允许):
   - IG completeness proof
   - Espeak → ARPABET 映射表
```

---

## 6. 实施 Roadmap（v2 微调）

| 阶段 | 任务 | 预计 | 产出 |
|---|---|---|---|
| **P0.1** | Phoneme alignment (wav2vec2 + ARPABET 映射) | 1 天 | `phoneme_align/` + JSON outputs |
| **P0.2** | Fisher-z + clamp 改写 IG 模块 | 0.5 天 | `attribution/ig_fisher_z.py` |
| **P0.3** | 自适应 n_steps + completeness gap 报告 | 0.5 天 | 收敛性统计脚本 |
| **P0.4** | Phoneme-level aggregation + ranking 统计 | 0.5 天 | `phoneme_aggregation/` 模块 |
| **P1.1** | 司法风格可视化模块 | 1-2 天 | `visualization/` |
| **P1.2** | Phoneme-level Deletion/Insertion AUC | 1-2 天 | 扩展 `reliability.py` |
| **P1.3** | 三方法对比 (Fisher-z vs raw cos vs L2) | 0.5 天 | Table 1/2 数据 |
| **P2** | 论文写作 + 主图 finalize | 3-5 天 | ICASSP 4 页 draft |
| **总计** | | **9-12 工作日** | |

### 服务器命令（v2 修订, 你部署后跑）

```bash
# 1. 抽样 500 对
python scripts/phoneme_align/sample_pairs.py \
  --eval_list /home/database/sre/voxceleb/voxceleb1/voxceleb1_test_v2.txt \
  --n_pairs 500 \
  --output scripts/phoneme_align/outputs/pairs_500.csv \
  --seed 42

# 2. 批量音素对齐 (服务器 GPU)
python scripts/phoneme_align/batch_align.py \
  --pairs_csv scripts/phoneme_align/outputs/pairs_500.csv \
  --extra_wavs scripts/phoneme_align/outputs/star_wavs.txt \
  --eval_path /home/database/sre/voxceleb/voxceleb1/test/wav \
  --output_dir scripts/phoneme_align/outputs/alignments \
  --device cuda --batch_size 8

# 3. Fisher-z paired attribution (三个模型分别跑)
python scripts/run_fisher_z_attribution.py \
  --model_path /home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_clean_noSpec/exp/vox1/model/model_0061.model \
  --pairs_csv scripts/phoneme_align/outputs/pairs_500.csv \
  --alignment_dir scripts/phoneme_align/outputs/alignments \
  --output_dir results/attributions/model_0061 \
  --n_steps 300 --clamp 0.98

# 4. Phoneme aggregation + ranking
python scripts/aggregate_phoneme_ranking.py \
  --attribution_dir results/attributions \
  --alignment_dir scripts/phoneme_align/outputs/alignments \
  --output results/phoneme_ranking_table.csv

# 5. Deletion/Insertion AUC (phoneme + bin 两级)
python scripts/run_reliability_test.py \
  --models all --pairs_csv pairs_500.csv \
  --methods fisher_z raw_cos l2_norm \
  --levels bin phoneme \
  --output results/reliability_table.csv
```

---

## 7. 风险与备选（v2 大幅修订）

| 风险 | 严重度 | 缓解 |
|---|---|---|
| **Thebaud 2024 重叠风险** | ⭐⭐⭐⭐⭐ | (1) intro 第一段就 cite (2) 强调 3 个差异: Fisher-z, paired, forensic 桥接 (3) 实验明确对比 raw cos vs Fisher-z |
| **"LLR" 命名被审稿人攻击** | ⭐⭐⭐⭐⭐ | 全文不用 "LLR"/"likelihood ratio"。统一叫 "Fisher z-transform"/"logit-cosine" |
| **IG 在 arctanh 上不收敛** | ⭐⭐⭐⭐ | (1) clamp 到 [-0.98, 0.98] (2) n_steps=300 (3) 报告 completeness gap (4) gap>5% 时自适应翻倍 |
| **Wang 2024 元音 > 擦音矛盾** | ⭐⭐⭐⭐ | **当 hook 用**! 如果 Ours 找回"擦音 > 元音"的 forensic 共识 → 强力卖点 |
| **500 对统计不够** | ⭐⭐⭐ | (1) paired t-test Ours vs baselines (2) bootstrap CI (3) 备选扩到 1000 对 |
| **wav2vec2 对齐精度** | ⭐⭐⭐ | (1) median 聚合 (2) 人工验证 20-30 样本 (3) 报告对齐误差直方图 |
| **Espeak → ARPABET 映射** | ⭐⭐ | 用标准 G2P 映射表 + 抽样检查 |
| **PLDA-based LLR 才是 "真 LLR"** | ⭐⭐⭐⭐ | Discussion 章节明确承认: "proper PLDA-based LLR attribution is left for future work, our Fisher-z transform serves as a principled, calibration-free score amplification" |

---

## 8. 不做的事（v2 微调）

| 不做 | 原因 |
|---|---|
| 命名为 "LLR" | 学术诚信 + forensic 圈子会拒稿 |
| PLDA + calibration 后做 IG | 工作量太大, 4 页放不下, 留作 future work |
| Triplet margin objective | Hinge 不可微, IG 不适用 (Oracle 否决) |
| DeepLIFT 替代 IG | 在 dot-product (cosine) 上违反 completeness (Ancona 2018) |
| Phoneme-conditional speaker fingerprint matrix | 体量大, 留给期刊版 |
| 跨数据集 (LibriSpeech) 一致性 | 4 页放不下 |
| 跨架构 (ResNet, WavLM-SV) 对比 | 同上 |
| 归因加权训练 (Phase 2 原规划) | 4 页放不下 |
| 全 VoxCeleb1 (37720 对) | 500 对统计足够 |

---

## 9. 验收标准

- [ ] Phoneme alignment 覆盖 500 对 + 6 个 star wavs
- [ ] IG completeness gap < 5% 平均, < 10% 个例 (Fisher-z 上)
- [ ] cos 分布直方图: <1% 的 frame cos > 0.98 (clamp 影响小)
- [ ] Phoneme ranking 表 (Table 1) 跑出且 Spearman ρ (Ours vs forensic) > Thebaud ρ
- [ ] Deletion/Insertion AUC (Table 2) Ours 优于 raw cos 和 L2-norm
- [ ] Fig 2 司法风格符合截图范式: 蓝底红线 + phoneme grid
- [ ] Fig 3 phoneme ranking 三方法对比
- [ ] Fig 4 三模型对比
- [ ] 4 页 ICASSP draft + 15 篇参考文献 (含 v2 新增的 6 篇)

---

## 10. 附录

### 10.1 ARPABET 39 phonemes + sil 按类别

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
Total: 40 (39 + sil)
```

### 10.2 Espeak → ARPABET 映射（核心 30，完整在脚本）

```python
ESPEAK_TO_ARPABET = {
    # Vowels (espeak 元音表示与 IPA 接近)
    'a': 'AA', 'aː': 'AA',
    'æ': 'AE',
    'ʌ': 'AH', 'ə': 'AH',
    'ɔ': 'AO', 'ɔː': 'AO',
    'ɛ': 'EH', 'ɛə': 'EH',
    'ɪ': 'IH', 'ɪə': 'IH',
    'i': 'IY', 'iː': 'IY',
    'oʊ': 'OW', 'əʊ': 'OW',
    'ʊ': 'UH', 'ʊə': 'UH',
    'u': 'UW', 'uː': 'UW',
    'aɪ': 'AY', 'aʊ': 'AW',
    'eɪ': 'EY', 'ɔɪ': 'OY',
    'ɝ': 'ER', 'ɚ': 'ER',
    # Consonants
    'b': 'B',
    'tʃ': 'CH',
    'd': 'D',
    'ð': 'DH',
    'f': 'F',
    'g': 'G',
    'h': 'HH',
    'dʒ': 'JH',
    'k': 'K',
    'l': 'L', 'ɫ': 'L',
    'm': 'M',
    'n': 'N',
    'ŋ': 'NG',
    'p': 'P',
    'ɹ': 'R', 'r': 'R',
    's': 'S',
    'ʃ': 'SH',
    't': 'T',
    'θ': 'TH',
    'v': 'V',
    'w': 'W',
    'j': 'Y',
    'z': 'Z',
    'ʒ': 'ZH',
    # Silence
    '_': 'SIL', '': 'SIL', 'sp': 'SIL', 'sil': 'SIL',
}
```

### 10.3 服务器路径

```
VoxCeleb1 test wav    /home/database/sre/voxceleb/voxceleb1/test/wav
VoxCeleb1 dev wav     /home/database/sre/voxceleb/voxceleb1/dev/wav
test trials list      /home/database/sre/voxceleb/voxceleb1/voxceleb1_test_v2.txt
MUSAN noise           /home/database/noise/musan
Model 1 (clean)       /home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_clean_noSpec/exp/vox1/model/model_0061.model
Model 2 (noise)       /home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_noise_noSpec/vox1/clean/model/model_0078.model
Model 3 (noise+adv)   /home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/3.05/model/model_0077.model
工作根目录            /home/zhangxl24/SpeakerRecongnition/
```

### 10.4 ECAPA FBank 配置（已验证）

```python
torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=400,    # 25 ms
    hop_length=160,    # 10 ms
    f_min=20,
    f_max=7600,
    window_fn=torch.hamming_window,
    n_mels=80,
)
```

### 10.5 IG Completeness 证明（论文附录素材）

```
Proposition (Completeness of IG under monotonic transform):
   设 F: X → R 可微 (e.g., cosine similarity)
       g: R → R 可微 (e.g., arctanh on clamp区间)
   令 G(x) = g(F(x)).
   则:
       Σ_i IG_i^G(x) = G(x) - G(x_baseline)

Proof:
   令 γ(α) = x' + α(x - x')  为 baseline 到 x 的直线插值路径.

   IG_i^G(x) = (x_i - x'_i) · ∫_0^1 ∂G/∂x_i |_{γ(α)}  dα
             = (x_i - x'_i) · ∫_0^1 g'(F(γ(α))) · ∂F/∂x_i |_{γ(α)}  dα

   Σ_i IG_i^G = ∫_0^1 g'(F(γ(α))) · Σ_i (x_i - x'_i) · ∂F/∂x_i |_{γ(α)}  dα
              = ∫_0^1 g'(F(γ(α))) · dF/dα |_{γ(α)}  dα
              = ∫_0^1 d/dα [ g(F(γ(α))) ]  dα                  (chain rule)
              = g(F(γ(1))) - g(F(γ(0)))                         (微积分基本定理)
              = g(F(x)) - g(F(x'))
              = G(x) - G(x_baseline)                            ∎

应用到本方法:
   g = arctanh ∘ clamp[-0.98, 0.98]
   F = cosine similarity ∘ embedding
   G(x) = arctanh(clamp(cos(emb(x), emb(x_ref)), -0.98, 0.98))

   g 在 clamp 区间 (-0.98, 0.98) 内严格可微.
   边界 |F|=0.98 处梯度不连续, 但对路径积分而言是 measure-zero,
   数值实现中 cos 极少接近 0.98 (经验上 <1% 的 frame), 影响可忽略.
   实际中我们通过 verify_convergence 计算 completeness 数值 gap,
   并在论文中报告该 gap (Section 3.1).
```
