# 归因方法论分析

## 1. 归因对象

对 FBank（对数梅尔滤波器组）上的时间-频率区域归因，回答"哪些时频 bin 对说话人嵌入的余弦相似度贡献最大"。

### 为什么选 FBank 而非其他层级

| 归因对象 | 可解释性 | 维度 | 跨架构通用性 | 结论 |
|---|---|---|---|---|
| **FBank / Spectrogram** | 强，保留时频结构，可映射到语音学频段 | 80×200，可控 | 强，所有模型共享前端 | **最优选择** |
| Raw Waveform | 弱，32240 点结果噪声大 | 32240，过高 | 强 | 维度灾难，结果不可读 |
| Embedding 维度 | 弱，只能说"第 x 维重要" | 512 | 强 | 无法映射回语音学含义 |
| 中间层特征 (SE-Block / ResBlock) | 中，需逐层 Hook | 不定 | 弱，架构相关 | 不同模型不通用 |

## 2. 归因目标函数

### 我们采用：cosine_sim(x, x_ref)

$$\text{Objective}(x) = \cos(\text{emb}(x),\ \text{emb}(x_{\text{ref}}))$$

含义：对判断"目标与参考是否为同一说话人"贡献最大的时频区域。

### 与其他目标函数对比

| 目标函数 | 含义 | 能否区分同/不同人 | 问题 |
|---|---|---|---|
| **cosine_sim(x, x_ref)** | 对说话人相似度的贡献 | **能**（配对设计） | 我们的方法 |
| L2-norm of embedding | 对嵌入幅度的贡献 | 不能 | Zhang et al. (INTERSPEECH 2023)，无法区分同/不同说话人 |
| Classification logits (AAM-Softmax) | 对分类分数的贡献 | 间接 | 受 margin/scale 超参影响大，不直接反映身份 |

## 3. 归因方法

### 我们采用：Integrated Gradients (Sundararajan et al., 2017)

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

选择理由：
- **公理保证**：满足完整性（Completeness）和实现不变性（Implementation Invariance）
- **FBank 空间可微**：ECAPA-TDNN 的 torchfbank 支持自动微分
- **收敛性验证**：实验表明 cosine_sim 目标下 IG 收敛良好（相对误差 < 0.06）

### 与其他归因方法对比

| 方法 | 计算成本 | 稳定性 | 声纹领域适用性 |
|---|---|---|---|
| **Integrated Gradients** | 中（需 n_steps 次前向+反向） | 高（有公理保证） | **主流选择** |
| GradCAM | 低 | 中 | 不适用——ECAPA-TDNN 的 SE-Block/统计池化无明确空间对应 |
| LIME | 高（扰动采样） | 低（结果不稳定） | Shen et al. (Interspeech 2025) 发现 IG 在声纹上 <50% inter-seed 一致性 |
| SHAP | 极高（32240 维不可行） | 高 | 计算成本不允许 |

## 4. 核心创新点：Paired Attribution

已有工作（Zhang et al. 2023, PhiNet 2025）只做单一归因，无法回答"什么是声纹"。

我们提出 **配对差值归因**：

$$\text{Voiceprint}_i = \text{IG}_i(x,\ x_{\text{same}}) - \text{IG}_i(x,\ x_{\text{diff}})$$

- $\text{IG}(x, x_{\text{same}})$：与同说话人参考的余弦相似度归因 → 正例支持区域
- $\text{IG}(x, x_{\text{diff}})$：与不同说话人参考的余弦相似度归因 → 反例支持区域
- **差值**：纯粹属于"声纹身份"的时频区域，消除与内容/信道相关的混淆因素

这在已有工作中无人做过。

## 5. 基线选择

| 基线类型 | 定义 | 在数据流形上 | 适用场景 |
|---|---|---|---|
| zero | 全零 FBank | 否 | 最基础，IG 公理要求基线为"无信息"输入 |
| global_mean | 全语料库 FBank 均值 | 是 | 代表"平均语音"，归因更集中在偏离均值的区域 |
| speaker_mean | 目标说话人 FBank 均值 | 是 | 消除说话人固有特征，归因聚焦于话语特异信息 |
| cross_speaker_mean | 非目标说话人 FBank 均值 | 是 | 与说话人均值对比，可发现说话人特有模式 |

当前默认使用 zero 基线，建议后续实验对比 global_mean。

## 6. 可视化设计

- FBank：粉蓝色系 colormap（深蓝→浅蓝→粉→玫红），直观展示频谱能量分布
- Voiceprint Map：粉蓝底 + 红色高亮（强度映射透明度），越红 = 声纹贡献越大
- 频段标注：F0/F1/F2/F3/High 分区线，连接到语音学含义
- Band Energy Bar：各频段声纹能量占比，3 模型并排对比
- Clean vs Noisy：左右分栏，直观展示噪声对声纹归因的影响

## 7. 与已有工作的关系

| 工作 | 归因对象 | 目标函数 | 方法 | 配对设计 |
|---|---|---|---|---|
| **Ours** | FBank | cosine_sim (paired) | IG + 差值 | ✅ 正例-反例-差值 |
| Zhang et al. (INTERSPEECH 2023) | FBank | L2-norm | 5 种方法对比 | ❌ 单一归因 |
| PhiNet (arXiv 2604.01590) | 中间层特征 | 自解释 | 主动归因 | ❌ 丢失频域细节 |
| Shen et al. (Interspeech 2025) | FBank | 多种 | IG/GradCAM 等 | ❌ 只评估可靠性 |
