# 引用文献证据库 (citations.md)

> 用途：所有 draft v2 中要引用的文献，**全部经过亲自 PDF 验证**。
> 标记说明：
> - **L1** 一手验证：我亲自读过该 PDF 的引用页码，原文摘录可贴入 draft
> - **L2** 二手转引：通过另一篇我读过的 PDF 转引（在 draft 中明确写 "via X"）
> - PDF 路径：`paper/references/pdfs/<filename>.pdf`
> - 引用 ID 用 `[Cite-NN]` 编号，draft 中直接用此编号

---

## A. 法证 - Practitioner Survey & Meta-Analysis

### [Cite-01] Fairclough, Brown & Kirchhübel 2023 (ICPhS) — L1 ★★★★★
- **完整引用**: Fairclough, L., Brown, G., & Kirchhübel, C. (2023). "Reviewing the performance of formants for Forensic Voice Comparison: a meta-analysis of forensic speech science research." *Proc. 20th Int. Congr. Phonetic Sciences (ICPhS 2023)*, Prague, paper ID 885.
- **PDF**: `ICPhS2023_Formant_MetaAnalysis.pdf` (5 pages)
- **机构**: Lancaster University + Soundscape Voice Evidence
- **关键 L1 摘录** (page 2 §2.2 "Formants in FVC casework"):
  > "Formants are widely used by forensic speech practitioners. In the survey by [1], it was found that of 36 experts, 97% of them carried out some form of formant analysis. Of the practitioners using formants, all measure F2, 87% measure F1 and F3, while 17% measure F4. Importantly, 94% measured the centre frequencies of monophthongs, 71% reported measuring the trajectories of diphthongs, and 45% measured vowel-consonant or consonant-vowel formant transitions [1]."
- **关键 L1 摘录** (page 5 §5 "Discussion"):
  > "It might come as a surprise to learn that relatively few studies use midpoint formant measurements in the speaker discrimination research literature... This overall low usage of midpoint measurements in the forensic speech science research literature seems to be at odds with the 97% of practitioners who reported to use midpoint measurements in forensic casework [1]."
- **数据**: 277 results from 37 papers, 1996-2021, 8 languages
- **draft 用途**: Section 2 (Forensic phonetics 共识 — 实务派 anchor)；Table 1 (Practitioner 列)
- **它转引的关键文献**: Gold & French 2011 [ICPhS ref 1]，R v O'Doherty 2002 [ICPhS ref 14]

---

### [Cite-02] (已删除 — Gold & French 2011 不再单独引用)
- **决策**: 用户已批准简化方案。Gold & French 2011 的关键数据（97% practitioners 使用 formant analysis 等）改为通过 **[Cite-01] ICPhS 2023 page 2 §2.2 转引承担**，draft 直接引 Cite-01 即可。
- **不再使用此 Cite ID** (保留编号位置避免后续重排)

---

### [Cite-03] Smorenburg & Heeren 2021 — L1 ★★★★★
- **完整引用**: Smorenburg, L., & Heeren, W. (2021). "Forensic value of acoustic-phonetic features from Standard Dutch nasals and fricatives." *Proc. AISV 2021*, Leiden University Centre for Linguistics.
- **PDF**: `Smorenburg_Heeren_2021_consonants.pdf` (2 pages)
- **关键 L1 摘录** (page 1 引言):
  > "Although vowels generally outperform consonants in speaker discrimination, reports indicate that forensic voice analysts regularly use consonants in auditory-acoustic analysis [1]." (其中 [1] = Gold & French 2011)
- **关键 L1 摘录** (page 1 引言, 转引 Rose 2002 p.135):
  > "nasal consonants exhibit high speaker-specificity because of the nature of a nasal; the involvement of the relatively rigid nasal cavity, which has different shapes and sizes between speakers, results in high between-speaker but low within-speaker variation for nasals [7, p.135]." ([7] = Rose 2002)
- **关键 L1 数据** (page 2 Table I, Median Cllr):
  | Phoneme | Static Cllr | Dynamic Cllr | Static nasal-spec Cllr |
  |---|---|---|---|
  | /s/ | **0.52** | 0.91 | — |
  | /x/ | 0.82 | 0.96 | — |
  | /n/ | 0.67 | 0.87 | **0.55** |
  | /m/ | 0.71 | 0.93 | 0.70 |
- **关键 L1 摘录** (page 2 Discussion):
  > "Against expectations, we found that /s/ outperformed the other consonants, even though it was sampled from telephone speech and its spectral peak falls outside of the telephone band."
- **draft 用途**: Section 2 量化派 anchor；Table 1 (Empirical Cllr 列)；论证 "vowels > nasals 不只是教科书也是数据"

---

### [Cite-04] Becker, Jessen & Grigoras 2008 — L1 ★★★★
- **完整引用**: Becker, T., Jessen, M., & Grigoras, C. (2008). "Forensic Speaker Verification Using Formant Features and Gaussian Mixture Models." *Proc. Interspeech 2008*, Brisbane.
- **PDF**: `Becker_2008_LTF_GMM.pdf` (4 pages)
- **机构**: Austrian Academy of Sciences + Bundeskriminalamt + Romanian Ministry of Justice
- **关键 L1 数据** (page 3 Table 1 — Equal error rates):
  | Features | EER |
  |---|---|
  | F1+F2 | 0.096 |
  | F2+F3 | 0.105 |
  | F1+F2+F3 | **0.053** |
  | F1+F2+BW1+BW2 | 0.042 |
  | F2+F3+BW2+BW3 | 0.060 |
  | F1+F2+F3+BW1+BW2+BW3 | **0.030** |
- **关键 L1 摘录** (page 3):
  > "The best performance with an equal error rate of 3% can be observed using all three formant frequencies and the corresponding bandwidths."
- **关键设置**: 68 male German speakers, telephone-bandwidth (8 kHz downsampled), spontaneous speech, formants extracted via WaveSurfer LPC
- **draft 用途**: Section 2 量化派 — vowel formants 在电话语音上 EER 3% 的硬数据

---

### [Cite-05] Cavalcanti, Eriksson, Barbosa & Madureira 2024 — L1 ★★★★★
- **完整引用**: Cavalcanti, J.C., Eriksson, A., Barbosa, P.A., & Madureira, S. (2024). "Revisiting the speaker discriminatory power of vowel formant frequencies under a likelihood ratio-based paradigm: The case of mismatched speaking styles." *PLOS ONE*, 19(12), e0311363. DOI: 10.1371/journal.pone.0311363
- **PDF**: `Cavalcanti_2024_PLOSONE.pdf` (27 pages)
- **关键 L1 摘录** (page 1 abstract):
  > "In order of relevance, F3, F4, and F1 performed the best in style-matched comparisons, as suggested by lower Cllr and EER values. F2 performed the worst intra-style in both Dialogue and Interview."
- **关键 L1 摘录** (page 1 abstract):
  > "The combination of higher formants 'F3 + F4' outperformed the combination of lower formants 'F1 + F2'."
- **关键 L1 摘录** (page 1 abstract):
  > "Front and central vowels were found more speaker discriminatory than back vowels as far as the 'F1 + F2' relation was concerned."
- **数据**: 20 male Brazilian Portuguese speakers, Dialogue + Interview style
- **draft 用途**: Section 2 — 量化数据支持 vowels 中 F3/F4 高判别力；前元音 > 后元音

---

## B. 法证 - 标准 / 法庭

### [Cite-06] ENFSI BPM 2021 — L1 ★★★★★
- **完整引用**: Wagner, I. (Ed.) (2021). "Best Practice Manual for the Methodology of Forensic Speaker Comparison." ENFSI-BPM-FSC-01, Version 01, ENFSI Forensic Speech and Audio Analysis Working Group, November 2021.
- **PDF**: `ENFSI_BPM_2021_FSC.pdf` (32 pages)
- **关键 L1 摘录** (page 7 §5.2 Acoustic Analysis):
  > "The analyses are conducted on both a segmental and supra-segmental basis. There is investigation into the articulation of vowels and consonants on the one hand, and on prosodic patterns like intonation on the other hand. Additionally, there are linguistic analyses of lexical and grammatical realisations in speech."
- **关键 L1 摘录** (page 8 §5.2.4 Formant Frequencies):
  > "Formants are the maxima of the vocal tract frequency response. They play a very important role in speech signal analysis as their positions are defined by the geometry and specific shape of the individual vocal tract of the speaker... Vocalised sounds are mostly used for formant analysis as they provide rich signal source spectrum."
- **draft 用途**: Section 2 — 欧洲官方标准明确将 vowel formants 作为法证核心声学参数

---

### [Cite-07] R v O'Doherty 2002 NICA 20 — L1 ★★★★★
- **完整引用**: *The Queen v Anthony O'Doherty* [2002] NICA 20. Northern Ireland Court of Criminal Appeal, Belfast, delivered 19 April 2002. Nicholson LJ.
- **PDF**: `RvODoherty_2002_NICA20.pdf` (29 pages)
- **关键 L1 摘录** (page 18, Nicholson LJ judgment, **法庭判决经典段**):
  > "Despite Mrs McClelland's expression of scepticism we are satisfied, having heard Dr Nolan and Dr French and read the report of Dr Kunzel, that in the present state of scientific knowledge **no prosecution should be brought in Northern Ireland in which one of the planks is voice identification given by an expert which is solely confined to auditory analysis. There should also be expert evidence of acoustic analysis** such as is used by Dr Nolan, Dr French and all but a small percentage of experts in the United Kingdom and by all experts in the rest of Europe, **which includes formant analysis**."
- **三个例外** (page 18): (i) 已知群体的声音归属；(ii) 罕见特征；(iii) 口音/方言问题。
- **draft 用途**: Section 2 — 法庭强制要求 formant analysis 的最强法理依据

---

### [Cite-08] King v Gleeson 2023 NICC 17 — L1 ★★
- **完整引用**: *King v Darren Gleeson* [2023] NICC 17. Crown Court in Northern Ireland, June 2023.
- **PDF**: `King_v_Gleeson_2023_NICC17.pdf`
- **draft 用途**: 可选 — 证明 O'Doherty 2002 在 2023 年仍是有效判例（仅 footnote 提及）
- **L1 状态**: 已下载未深读，仅作 supporting evidence

---

## C. 法证 - 综述 & PhD 论文 (二手参考)

### [Cite-09] Morrison & Enzinger 2019 — L1 ★★★
- **完整引用**: Morrison, G.S. & Enzinger, E. (2019). "Introduction to forensic voice comparison." In *The Routledge Handbook of Phonetics* (W.F. Katz & P.F. Assmann, Eds.). London: Routledge.
- **PDF (preprint)**: `MorrisonEnzinger_2019_FVC_intro.pdf` (51 pages preprint)
- **draft 用途**: Section 2 综述背景 — 区分四种 FVC 方法（auditory / spectrographic / acoustic-phonetic / automatic）
- **L1 状态**: 已下载，关键综述章节可作 "via" 转引来源

---

### [Cite-10] French 2017 — L1 ★★
- **完整引用**: French, J.P. (2017). "A developmental history of forensic speaker comparison in the UK." *English Phonetics*, 271-286.
- **PDF**: `French_2017_UK_FSC_history.pdf` (15 pages)
- **draft 用途**: 英国法证发展史背景，可选

---

### [Cite-11] Kavanagh 2012 York PhD — L1 ★★★
- **完整引用**: Kavanagh, C. (2012). "New consonantal acoustic parameters for forensic speaker comparison." PhD thesis, University of York. White Rose etheses ID 3980.
- **PDF**: `Kavanagh_2012_York_PhD_consonants.pdf` (427 pages)
- **draft 用途**: Section 2 — /m, n, ŋ, l, s/ 系统性法证评估，可作 nasal 实证数据来源
- **L1 状态**: 已下载，章节级深读待 draft 实际需要时进行

---

### [Cite-12] Hughes 2014 York PhD — L1 ★★
- **完整引用**: Hughes, V. (2014). "The definition of the relevant population and the collection of data for likelihood ratio-based forensic voice comparison." PhD thesis, University of York. eprint ID 8309.
- **PDF**: `Hughes_2014_York_PhD_LR.pdf`
- **draft 用途**: LR-based FVC 方法学背景

### [Cite-13] Gold 2014 York PhD — L1 ★★
- **完整引用**: Gold, E.A. (2014). "Calculating likelihood ratios for forensic speaker comparisons using phonetic and linguistic parameters." PhD thesis, University of York. eprint ID 6166.
- **PDF**: `Gold_2014_York_PhD_LR.pdf`
- **draft 用途**: 可作 Gold & French 2011 方法学的扩展参考

### [Cite-14] Holmes 2023 York PhD — L1 ★★
- **完整引用**: Holmes, E. (2023). "Towards Phonetically-Informed Automatic Speaker Recognition." PhD thesis, University of York. eprint ID 37020.
- **PDF**: `Holmes_2023_York_PhD_phoneme_ASR.pdf` (265 pages)
- **draft 用途**: phonetic-informed ASR 直接相关方向

---

## D. Neural SV - Interpretability

### [Cite-15] PhiNet (Ma, Wang, Liu, Li 2026) — L1 ★★★★★
- **完整引用**: Ma, Y., Wang, S., Liu, T., & Li, H. (2026). "PhiNet: Speaker Verification with Phonetic Interpretability." arXiv:2604.01590 (also IEEE TASLP submission per LaTeX template).
- **PDF**: `PhiNet.pdf` (14 pages)
- **机构**: National University of Singapore + Nanjing University + LIGHTSPEED
- **关键 L1 摘录** (page 2 Introduction):
  > "our model consistently identifies **nasals and vowels** as crucial for speaker verification, while also capturing significant variability in the distinctiveness of fricative sounds."
- **关键 L1 摘录** (page 2):
  > "this global interpretation aligns with human perception in FSC, as nasals and vowels (e.g., 'N', 'M', and 'AY') are widely recognized as more discriminative for speaker verification tasks."
- **关键 L1 数据** (page 8 Fig 5 — Phonetic weight ranking, 训练自学得):
  - 最高: N, S, M, AY (vowels + nasals + sibilants)
  - 最低: B, P, HH (stops + glottal fricative)
- **关键 L1 数据** (page 10 Table II — System (10) PhiNet 3s vs ECAPA-TDNN 3s on Vox1-O):
  | System | EER (%) | minDCF |
  |---|---|---|
  | ECAPA-TDNN (reproduced, 3s, Vox1) | 3.541 | 0.279 |
  | PhiNet (System 10, 3s, Vox1) | 5.673 | 0.439 |
- **关键 L1 摘录** (page 8 Fig 6 caption + 文字):
  > "Across all configurations, the ranking of phonetic weights remains relatively similar, with phonemes like 'N', 'S', and 'M' consistently contributing the most."
- **架构**: ECAPA-TDNN backbone + per-phoneme average pooling + 学习的 phonetic weight $w_i \in [0,1]$ via min-max norm
- **训练设置**: VoxCeleb1 / VoxCeleb2, 3-second segments optimal, MUSAN + RIR augmentation
- **draft 用途**: Related Work — **互补不矛盾**：PhiNet 改架构能用上 nasals；我们的标准 ECAPA 不能用 nasals

---

### [Cite-16] Wang & Li 2024 实际 = Li, Wang, Li, Hamdulla, Wang 2024 — L1 ★★★★★
- **完整引用**: Li, P., Wang, T., Li, L., Hamdulla, A., & Wang, D. (2024). "How Phonemes Contribute to Deep Speaker Models?" arXiv:2402.02730. Also published in *2024 IEEE ICASSPW*, pp. 838-842.
- **PDF**: `Wang_Li_2024_phoneme_contrib.pdf` (6 pages)
- **机构**: Xinjiang University + BUPT + Tsinghua
- **关键 L1 摘录** (page 1 abstract):
  > "(1) At the population level, **vowels are more important than consonants**, confirming the human perception studies. However, fricatives are among the most unimportant phonemes, which contrasts with previous studies. (2) At the speaker level, a large between-speaker variation is observed regarding phoneme importance, indicating that whether a phoneme is important or not is largely speaker-dependent."
- **关键方法** (page 2): LayerCAM + Time Align Occlusion (TAO)，依据 LayerCAM 是 CAM 家族中唯一通过 reliability check 的方法
- **draft 用途**: Section 2 + Related Work — **与我们结论一致**的论文：vowels > consonants in deep SV
- **⚠️ 重要修正**: 之前 librarian 标为 "Wang & Li 2024" 是错误的。**正确通用 cite 名应该是 "Li et al. 2024"**

---

### [Cite-17] Thebaud, Hernández, Samson Juan & Tahon 2024 — L1 ★★★★★
- **完整引用**: Thebaud, T., Hernández, G., Samson Juan, S.F., & Tahon, M. (2024). "A Phonetic Analysis of Speaker Verification Systems through Phoneme Selection and Integrated Gradients." *Proc. Speaker and Language Recognition Workshop (Odyssey 2024)*.
- **PDF**: `Thebaud_2024_Odyssey_phoneme_IG.pdf` (8 pages)
- **机构**: JHU CLSP + CENATAV Cuba + UNIMAS Malaysia + LIUM Le Mans
- **关键 L1 摘录** (page 1 abstract):
  > "We use Whisper and the Montreal Forced Aligner (MFA) to transcribe, then segment phonetically the VoxCeleb1 test set. Phoneme selection is first used, before computation of the x-vectors, to observe which phonemes are the most discriminative through their impact on EER and MinDCF metrics. Integrated Gradients are then used to show which phonemes yielded the highest gradients comparing two speakers."
- **关键 L1 摘录** (page 1 abstract):
  > "We find that for the representation of the x-vector in speaker recognition systems, **both consonants and vowels are relevant and important** to capture the distinctive characteristics of a speaker's voice."
- **关键 L1 摘录** (page 2 §2.1):
  > "Many works in the literature provide clues that phoneme categories such as vowels and nasals are discriminant for speaker identification [15], but specific phonemes such as /s/ also get high speaker verification performances. In another work, fricatives and stops consonants such as /s/, /t/ or /b/ have been found to perform worse than vowels and nasal [16]."
- **draft 用途**: Method 部分对比对象 — 第一篇把 IG 用于 SV pairwise 的工作（但 single attribution，非 paired）

---

### [Cite-18] ExPO (Ma, Wang, Liu, Li 2025) — L1 ★★★
- **完整引用**: Ma, Y., Wang, S., Liu, T., & Li, H. (2025). "ExPO: Explainable Phonetic Trait-Oriented Network for Speaker Verification." *IEEE Signal Processing Letters*. arXiv:2501.05729.
- **PDF**: `ExPO_Ma_2025.pdf` (5 pages)
- **关键 L1 摘录** (page 1 abstract):
  > "ExPO not only generates utterance-level speaker embeddings but also allows for fine-grained analysis and visualization of phonetic traits, offering an explainable speaker verification process."
- **draft 用途**: PhiNet 前身，提及即可

---

### [Cite-19] Ben-Amor (BA-LR) 2023/2024 — L1 ★★★
- **完整引用 (Interspeech 2023)**: Ben-Amor, I., Bonastre, J.-F., O'Brien, B., & Bousquet, P.-M. (2023). "Describing the phonetics in the underlying speech attributes for deep and interpretable speaker recognition." Interspeech 2023.
- **完整引用 (Odyssey 2024)**: Ben-Amor, I., et al. (2024). Odyssey 2024.
- **PDFs**: `BA-LR_Interspeech2023.pdf` + `BA-LR_Odyssey2024.pdf`
- **draft 用途**: Related Work — Binary attribute LR decomposition framework
- **L1 状态**: 已下载，仅 abstract 验证

---

### [Cite-20] Zhang et al. 2023 INTERSPEECH — L1 (已有原文) ★★★
- **完整引用**: Zhang, J., He, L., Guo, X., & Ma, J. (2023). "A study on visualization of voiceprint feature." *Proc. INTERSPEECH 2023*, pp. 2233-2237.
- **PDF**: `Zhang_2023_INTERSPEECH_visualization.pdf`
- **draft 用途**: Related Work — voiceprint visualization 中文学者工作（PhiNet ref [66] 也引）

### [Cite-22] Xu, Plumbley & Wang 2026 (SV XAI 评估) — L1 ★★★★ 【并发工作 concurrent】
- **完整引用**: Xu, Y., Plumbley, M. D., & Wang, W. (2026). "Explainable AI in Speaker Recognition — Attention Map Visualisation and Evaluation." arXiv:2606.22901 (eess.AS, submitted 22 Jun 2026, work in progress).
- **PDF**: `Xu_2026_SV_XAI_RISE.pdf` (9 pages)
- **机构**: CVSSP, University of Surrey
- **归因目标**: speaker **identification** 身份（摘要明言 "trained to identify speaker identity from a given utterance"）= 固定单一 ID 目标，不变归因目标。
- **方法**: 对比 GradCAM vs LayerCAM 两种算法；提出 Modified RISE-eval 评估注意力图质量。结论：GradCAM/LayerCAM 在不同实验条件下各有优势。
- **draft 用途**: Related Work — **并发工作**（早一个月，同做 SV XAI + LayerCAM）。必引并区分：**它固定 identification 目标、比较算法与图评估方法；我们把归因目标本身当研究变量、用 verification-native 目标 + 因果删除**。支撑我们“现有工作固定单目标只比算法”的现状描述。
- ⚠️ **并发声明**：2026-06-22 arXiv，work in progress，时间线与本文擞。引时标 concurrent，不得声称对方拄我们工作。

---

## E. 工具与 Backbone

### [Cite-21] Desplanques, Thienpondt & Demuynck 2020 (ECAPA-TDNN) — L1 ★★★★
- **完整引用**: Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification." *Proc. Interspeech 2020*. arXiv:2005.07143.
- **PDF**: `ECAPA-TDNN_Desplanques_2020.pdf` (5 pages)
- **机构**: IDLab, Ghent University
- **关键 L1 摘录** (page 2 §3.1 attentive pooling):
  > "we extend this temporal attention mechanism even further to the channel dimension. This enables the network to focus more on speaker characteristics that do not activate on identical or similar time instances, e.g. **speaker-specific properties of vowels versus speaker-specific properties of consonants**."
- **关键架构**: Channel- and context-dependent statistics pooling (Eq.1 channel attention)
- **draft 用途**: Section 3 backbone 介绍；**关键引文**：作者自己承认 attentive pooling 处理 vowels vs consonants 是设计目标之一 → 支持我们的"duration bias 来自 attentive pooling"假设

---

### [Cite-22] Sundararajan, Taly & Yan 2017 (Integrated Gradients) — L1 ★★★★
- **完整引用**: Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." *Proc. ICML 2017*. arXiv:1703.01365.
- **PDF**: `IntegratedGradients_Sundararajan_2017.pdf` (11 pages)
- **关键 L1 摘录** (page 1 abstract):
  > "We identify two fundamental axioms—**Sensitivity and Implementation Invariance** that attribution methods ought to satisfy."
- **draft 用途**: Section 3 — IG 方法基础引用

---

### [Cite-23] Nagrani, Chung & Zisserman 2017 (VoxCeleb1) — L1 ★★★
- **完整引用**: Nagrani, A., Chung, J.S., & Zisserman, A. (2017). "VoxCeleb: A Large-Scale Speaker Identification Dataset." *Proc. Interspeech 2017*. arXiv:1706.08612.
- **PDF**: `VoxCeleb1_Nagrani_2017.pdf`
- **draft 用途**: 数据集来源

### [Cite-24] Chung, Nagrani & Zisserman 2018 (VoxCeleb2) — L1 ★★
- **完整引用**: Chung, J.S., Nagrani, A., & Zisserman, A. (2018). "VoxCeleb2: Deep Speaker Recognition." *Proc. Interspeech 2018*. arXiv:1806.05622.
- **PDF**: `VoxCeleb2_Chung_2018.pdf`
- **draft 用途**: 数据集来源（仅 footnote）

---

### [Cite-25] Baevski et al. 2020 (wav2vec 2.0) — L1 ★★
- **完整引用**: Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *NeurIPS 2020*. arXiv:2006.11477.
- **PDF**: `wav2vec2_Baevski_2020.pdf` (19 pages)
- **draft 用途**: phoneme alignment 方法基础

### [Cite-26] Xu et al. 2021 (wav2vec2-Phoneme) — L1 ★★★
- **完整引用**: Xu, Q., Baevski, A., & Auli, M. (2021). "Simple and Effective Zero-shot Cross-lingual Phoneme Recognition." arXiv:2109.11680.
- **PDF**: `wav2vec2_Phoneme_Xu_2021.pdf` (8 pages)
- **draft 用途**: phoneme alignment — 我们使用的 wav2vec2-lv-60-espeak-cv-ft model

---

## F. 已决定不使用 (用户批准的简化方案)

> 以下文献付费墙且较老/可替代，决定不在 draft 中引用。
> 关键数据通过已下载的 L1 文献承担。

| 原计划引用 | 处理方案 | 替代来源 |
|---|---|---|
| Gold & French 2011 IJSLL | ❌ 不引 | [Cite-01] ICPhS 2023 转引 |
| Gold & French 2019 IJSLL | ❌ 不引 | 不需要 |
| Morrison et al. 2016 INTERPOL | ❌ 不引 | [Cite-01]+[Cite-06]+[Cite-07] |
| Hughes 2017 filled pause | ❌ 不引 | [Cite-12] Hughes 2014 PhD |
| McDougall 2004 双元音 | ❌ 不引 | 我们不做 dynamic formant，不需要 |
| Morrison 2009 JASA DCT | ❌ 不引 | 同上 |
| Amino & Arai 2009 FSI 鼻音 | ❌ 不引 | [Cite-03]+[Cite-11] |

### [Cite-27] Amino & Osanai 2012 — 已下载但不使用
- **完整引用**: Amino, K. & Osanai, T. (2012). "Speaker characteristics that appear in vowel nasalisation and their change over time." *Acoustical Science and Technology*, 33(2), 96-103.
- **PDF**: `Amino_Arai_2012_AST_nasals.pdf` (10 pages)
- **⚠️ 重要修正**: 之前 librarian 误标为 "Amino & Arai 2012 nasal acoustics"。实际是 Amino & **Osanai** (不是 Arai)，且是关于 **vowel nasalisation** 而非 nasal consonants。**与我们主题不直接相关，不引用。**

---

## 待办事项 (开始写 draft v2 时再做)

- 章节级深读 Kavanagh 2012 PhD (427 pages) 以获取 nasal 系统比较的 Cllr/EER
- 章节级深读 Hughes 2014 PhD (relevant population data)
- 验证 Holmes 2023 PhD 与我们的方向是否过于接近（可能需要避免抢点）
- 如果引用 ENFSI BPM Tippett plot 章节，需要再深读
- 如果引用 PhiNet Fig 7 leave-one-out 实验，需 page 9 再细读

---

## 引用使用规则 (draft v2)

1. **引用每个 fact 时必须带 [Cite-NN]**，且 NN 必须在本文档中存在
2. **L1 引用**：可以直接放原文段落（带引号），需在 draft 中注明页码
3. **L2 引用**：必须写 "via [Cite-XX]"，或在 reference list 标注 "secondary citation"
4. **数字必须从 L1 取**，不能凭印象写（如 "97%" 必须从 ICPhS 2023 page 2 取，不能从 librarian 报告抄）
5. **本 citations.md 之外的任何引用 → 拒绝写入 draft**
