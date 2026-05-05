# 归因可靠性验证实验方案

## 一、背景

Zhang et al. (INTERSPEECH 2023) 使用归因加权训练 + EER 对比来验证归因可靠性。
我们的方法与他们的核心差异在于归因目标：我们使用 **cosine_sim(同说话人) - cosine_sim(不同说话人)** 直接度量区分性，
而非 embedding 各维度平均。因此验证逻辑需要适配。

## 二、实验1：归因加权训练（与 Zhang et al. 对齐）

### 原理
如果归因正确识别了声纹特征，那么用归因加权后的 FBank 训练，性能应接近原始训练。

### 流程
1. 对训练集每个样本计算归因（IG + cosine_sim_diff / IG + L2-norm）
2. 生成加权训练数据：
   - 方案A: element-wise 乘法  `FBank_weighted = A ⊙ FBank`
   - 方案B: 阈值 mask  `FBank_masked = FBank * (A > threshold)`
3. 在加权数据上 fine-tune ECAPA-TDNN
4. 在 VoxCeleb1 标准测试集上评估 EER

### 对照组

| 组别 | 训练数据 | 预期 |
|------|---------|------|
| A. 原始训练 | FBank（无归因） | EER基准 |
| B. Ours (cosine_sim) 加权 | A_cosine ⊙ FBank | EER接近A → 可靠 |
| C. L2-norm 加权 | A_l2 ⊙ FBank | EER对比 |
| D. Ours + mask(0.1) | FBank * (A_cosine > 0.1) | EER略高但可接受 |
| E. Ours + mask(0.3) | FBank * (A_cosine > 0.3) | EER更高，看信息损失 |
| F. Random mask | FBank * (random > threshold) | EER大幅上升 → 证明不是随机 |

### 判断标准
- B ≈ A → 归因保留了几乎所有声纹信息，可靠
- B > A 但 < F → 归因保留大部分信息，有少量损失
- B ≈ F → 归因不可靠

### 备注
全量训练较慢，建议从预训练模型 fine-tune 几个 epoch。此实验为 Phase 2。

---

## 三、实验2：特征删除/插入测试（Deletion/Insertion AUC）⭐优先执行

### 原理
如果归因正确标注了声纹频段，那么：
- **Deletion**: 删除高归因频段 → 性能大幅下降（比随机删除降得更多）
- **Insertion**: 仅恢复高归因频段 → 性能快速恢复（比随机恢复快得多）

### 优势
- 不需要重新训练，直接在预训练模型上评估
- 几小时即可出结果
- 可视化效果好（曲线图）

### 流程

#### Step 1: 计算归因
对测试集中每对 enrollment/test 语音计算归因：
- Ours: IG + cosine_sim_diff
- Baseline: IG + L2-norm
- 得到 attribution map A ∈ R^{80×T}

#### Step 2: Deletion Test
按 attribution 值从高到低，逐步删除（置零）FBank 的频段：
- 删除 top-5% 归因频段 → 计算 cosine_sim → 记录
- 删除 top-10% → 计算 cosine_sim → 记录
- 删除 top-20% → ...
- 删除 top-50% → ...
- 计算 EER 随删除比例的变化曲线

#### Step 3: Insertion Test
从全零输入开始，按 attribution 值从高到低逐步恢复：
- 保留 top-5% 归因频段，其余置零 → 计算 cosine_sim → 记录
- 保留 top-10% → ...
- 保留 top-20% → ...
- 计算 EER 随保留比例的变化曲线

#### Step 4: Random 基线
同样做 deletion/insertion，但随机选择频段（不按归因值排序），多次取平均。

### 关键指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **Deletion AUC** | 删除曲线下面积 | 越大 = 归因越精准（删少量就导致性能崩溃） |
| **Insertion AUC** | 插入曲线下面积 | 越大 = 归因越精准（恢复少量就能恢复性能） |
| **Random AUC** | 同上，随机删/插 | 对比参照 |

### 对比维度
- Ours (cosine_sim_diff) vs L2-norm
- 3个模型分别做
- 频段维度删除 vs 时间维度删除（看声纹更依赖频率还是时间）

### 预期结果
1. **Deletion**: 删除 top-5% cosine_sim 归因频段 → EER 大幅上升（比 random 删除上升更多）
2. **Insertion**: 仅保留 top-20% cosine_sim 归因频段 → EER 已接近原始水平
3. **cosine_sim > L2-norm**: cosine_sim 的 deletion/insertion AUC 都优于 L2-norm
4. **对抗训练模型更鲁棒**: Noise_adv_vox1 模型的归因更集中在关键频段

### 论文表述模板
> "Our cosine_sim paired attribution achieves X% higher Deletion-AUC than L2-norm baseline,
> confirming that the identified features are genuinely discriminative for speaker identity."

---

## 四、执行计划

```
Phase 1（快速验证）: 实验2 — Deletion/Insertion Test
  → 确认归因方法基本可靠
  → 对比 cosine_sim vs L2-norm
  → 1-2天可完成

Phase 2（完整验证，如需要）: 实验1 — 归因加权训练
  → 与 Zhang et al. 的 Table 2 完全对齐
  → 需要 fine-tune，3-5天
```
