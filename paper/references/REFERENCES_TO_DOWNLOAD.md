# 论文引用文献完整清单

> 用途：论文 draft v2 所需全部引用文献的清单与下载状态追踪
> 原则：所有 draft 引用必须先在此清单中列出 → 下载 / 二手转引 → 进入 citations.md → 才能在 draft 中使用
> 避免 LLM 幻觉：未亲自验证的引用不写

## 状态标记说明

- `[OA]` 开放获取，可直接下载 PDF
- `[OA-Preprint]` 有 arXiv / 机构存档预印本
- `[Paywall]` 付费墙，正式发表版需要订阅
- `[Book]` 书籍，一般无法在线下载
- `[Court]` 法庭判决书，通常官网公开
- `[Standard]` 行业标准文档（如 ENFSI BPM），通常官网公开

## 优先级说明

- `[P0]` 论文 Related Work / Table 1 anchor 文献，**必须有原文或可靠二手转引**
- `[P1]` 重要佐证文献，**强烈建议**有原文
- `[P2]` 补充文献，二手转引即可

---

## A. Forensic Phonetics — 综述论文

### A1. Jessen 2008 "Forensic Phonetics" [P2] [Paywall]
- **作者**: Jessen, M.
- **出版**: *Language and Linguistics Compass*, 2(4), 671-711
- **DOI**: 10.1111/j.1749-818X.2008.00066.x
- **draft 用途**: forensic phonetics 综述
- **下载状态**: ❌ Paywall (Wiley)

> **注**：原 A1-A2（Rose 2002 / Nolan 1983）和 A4（Hollien）为书籍，按用户要求已从清单移除。
> 若需要他们的核心论证（如"鼻腔解剖刚性"），将通过 Smorenburg & Heeren 2021 §1 转引位置引用。

---

## B. Forensic — Practitioner Survey (实务派核心)

### B1. Gold & French 2011 "International practices in forensic speaker comparison" [P0] [Paywall]
- **作者**: Gold, E. & French, P.
- **出版**: *International Journal of Speech, Language and the Law*, 18(2), 293-307 (2011)
- **DOI**: 10.1558/ijsll.v18i2.293
- **关键数据**:
  - 100% 从业者做 vowel 分析
  - 97% 做 formant analysis
  - 100% 测 F2, 86% 测 F1/F3
  - 33% 认为 voice quality 最有用
  - 29% 认为 vowel formants 最有用
  - Nasals Likert 4.08/6 vs Fricatives 4.85/6
- **draft 用途**: Section 2.1 Table 1 (Practitioner 列) 最重要 anchor
- **下载状态**: ❌ Paywall (Equinox)
- **机构存档检查**: 检查 York 大学（Gold 当时机构）eprints — 待验证
- **二手转引**: 我们之前调研的 librarian 报告已有 verbatim 数据

### B2. Gold & French 2019 "International practices in forensic speaker comparisons: Second survey" [P1] [Paywall]
- **作者**: Gold, E. & French, J. P.
- **出版**: *International Journal of Speech, Language and the Law*, 26(1), 1-20 (2019)
- **DOI**: 10.1558/ijsll.38028
- **关键数据**:
  - ASR 使用从 17% (2011) 升到 41.2% (2019)
  - 验证 2011 结论的趋势变化
- **draft 用途**: Section 2.1 趋势分析
- **下载状态**: ❌ Paywall

### B3. Morrison et al. 2016 "INTERPOL survey" [P1] [Paywall]
- **作者**: Morrison, G.S., Sahito, F.H., Jardine, G., Djokic, D., Clavet, S., Berghs, S., Goemans Dorny, C.
- **出版**: *Forensic Science International*, 263, 92-100 (2016)
- **DOI**: 10.1016/j.forsciint.2016.03.044
- **关键数据**: 44 国 INTERPOL 调查，auditory-acoustic-phonetic 是欧洲最普遍
- **draft 用途**: Section 2.1 跨国实务对比
- **下载状态**: ❌ Paywall (Elsevier)
- **预印本检查**: Morrison 个人网站 forensic-voice-comparison.net 可能有 — 待验证

### B4. ICPhS 2023 Meta-analysis (Formant performance) [P0] [OA]
- **出版**: *Proc. 20th Int. Congr. Phonetic Sciences (ICPhS 2023)*, Prague, paper 885
- **关键数据**:
  - 37 项研究 277 个结果元分析
  - 97% practitioners 使用 midpoint formant measurements
  - Research literature vs practitioner practice 的 gap
- **draft 用途**: Section 2.1 + 2.3 实证支持
- **下载链接**: https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2023/full_papers/885.pdf
- **下载状态**: ✅ 可下载

---

## C. Forensic — Empirical EER/Cllr Studies (量化派核心)

### C1. Smorenburg & Heeren 2021 "Forensic value of consonants: Fricatives and nasals in Dutch" [P0] [OA]
- **作者**: Smorenburg, L. & Heeren, W.
- **出版**: Proc. AISV 2021
- **关键数据**:
  - 直接引用 "vowels generally outperform consonants"
  - 荷兰语 /n/ Cllr 0.55, /s/ Cllr 0.52
  - 转引 Rose 2002 p.135 鼻腔刚性
- **draft 用途**: Section 2.2 Table 1 (Empirical 列) 最重要 anchor
- **下载链接**: http://www.cl.uzh.ch/dam/jcr:71ca8375-fcc1-496f-b49c-298bd462cd62/AISV_2021_Smorenburg%20Heeren.pdf
- **下载状态**: ✅ 可下载 (uzh.ch 公开)

### C2. Becker et al. 2008 "Forensic Speaker Verification Using Formant Features and GMM" [P0] [OA]
- **作者**: Becker, T., Jessen, M., Grigoras, C.
- **出版**: Proc. Interspeech 2008
- **关键数据**:
  - F1+F2+F3+BW1+BW2+BW3 → EER 3%
  - F1+F2+F3 → EER 5.3%
- **draft 用途**: Section 2.2 vowel LTF 高判别力证据
- **下载链接**: https://www.isca-archive.org/interspeech_2008/becker08_interspeech.pdf
- **下载状态**: ✅ ISCA archive 公开

### C3. Hughes, Foulkes, Wood 2017 "Strength of FVC evidence from filled pauses" [P1] [Paywall]
- **作者**: Hughes, V., Foulkes, P., Wood, S.
- **出版**: *International Journal of Speech, Language and the Law*, 24(2), 211-236
- **DOI**: 10.1558/ijsll.32616
- **关键数据**: filled-pause /um/ 元音+鼻音组合 EER 4.08%, Cllr 0.12
- **draft 用途**: Section 2.2 元音+鼻音组合证据
- **下载状态**: ❌ Paywall
- **预印本检查**: York eprints — 待验证

### C4. McDougall 2004 "Speaker-specific formant dynamics: An experiment on Australian English /aɪ/" [P1] [Paywall]
- **作者**: McDougall, K.
- **出版**: *International Journal of Speech, Language and the Law*, 11(1), 103-130
- **DOI**: 10.1558/sll.2004.11.1.103
- **关键数据**: F1+F2+F3 双元音轨迹分类准确率 88-95%
- **draft 用途**: Section 2.2 diphthong dynamic 证据
- **下载状态**: ❌ Paywall
- **预印本检查**: Cambridge / York eprints — 待验证

### C5. Morrison 2009 "LR-based FVC using DCT of diphthong trajectories" [P1] [Paywall]
- **作者**: Morrison, G.S.
- **出版**: *Journal of the Acoustical Society of America*, 125(4), 2387-2397
- **DOI**: 10.1121/1.3081384
- **关键数据**: 5 双元音 DCT Cllr 0.095
- **下载状态**: ❌ Paywall (ASA)
- **预印本检查**: Morrison 个人网站 forensic-voice-comparison.net — 大概率有

### C6. Cavalcanti et al. 2024 "Revisiting speaker discriminatory power of vowel formant frequencies" [P0] [OA]
- **作者**: Cavalcanti, J.C., et al.
- **出版**: *PLOS ONE*, 19(12), e0311363
- **DOI**: 10.1371/journal.pone.0311363
- **关键数据**:
  - F3 > F4 > F1 > F2 判别力排名
  - 前元音 > 后元音
  - 风格不匹配影响讨论
- **draft 用途**: Section 2.2 元音 formant 判别力
- **下载链接**: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0311363&type=printable
- **下载状态**: ✅ PLOS ONE OA

### C7. Kavanagh 2012 PhD Thesis "An evaluation of /m, n, ŋ, l, s/ as forensic-phonetic parameters" [P2] [OA]
- **作者**: Kavanagh, C.
- **机构**: University of York PhD thesis
- **下载链接**: https://etheses.whiterose.ac.uk/3744/
- **下载状态**: ✅ 大概率可下 (White Rose etheses)

### C8. Amino & Arai 2009 "Speaker-dependent characteristics of the nasals" [P2] [Paywall]
- **出版**: *Forensic Science International*, 185(1-3), 21-28
- **DOI**: 10.1016/j.forsciint.2008.12.005
- **下载状态**: ❌ Paywall (Elsevier)

### C9. Amino & Arai 2012 "Acoustic analysis of nasality for forensic speaker comparison" [P2] [OA]
- **出版**: *Acoust. Sci. Technol.*, 33(2), 96-103
- **下载链接**: https://www.jstage.jst.go.jp/article/ast/33/2/33_2_96/_pdf
- **下载状态**: ✅ J-STAGE 公开

---

## D. Forensic — Standards and Legal

### D1. ENFSI BPM 2021 [P0] [Standard] [OA]
- **完整名称**: Wagner, I. (Ed.) "Best practice manual for the methodology of forensic speaker comparison"
- **机构**: ENFSI Forensic Speech and Audio Analysis Working Group (FSAAWG)
- **版本**: FSA-BPM-003, 2021-07
- **关键数据**: LTFD 方法、formant analysis 要求
- **draft 用途**: Section 2.1 欧洲标准
- **下载链接**: https://enfsi.eu/wp-content/uploads/2021/07/2021-07-07-final-draft-BPM-SPEAKER-COMPARISON.pdf
- **下载状态**: ✅ ENFSI 官网公开

### D2. R v O'Doherty [2002] NICA 20 [P0] [Court] [OA]
- **完整名称**: Northern Ireland Court of Criminal Appeal, Belfast, April 2002
- **关键数据**: 北爱尔兰强制要求 formant analysis
- **draft 用途**: Section 2.1 法庭约束力证据
- **下载链接**: https://www.judiciaryni.uk/files/judiciaryni/decisions/Queen%20v%20Anthony%20O%27Doherty.pdf
- **下载状态**: ✅ 法庭官网公开

### D3. R v Flynn & St John [2008] EWCA Crim 970 [P2] [Court] [OA]
- **关键数据**: 英格兰 & 威尔士不跟随 O'Doherty
- **下载链接**: https://www.bailii.org/ew/cases/EWCA/Crim/2008/970.html
- **下载状态**: ✅ BAILII 公开 (HTML, 非 PDF)

### D4. King v Gleeson [2023] NICC 17 [P2] [Court] [OA]
- **关键数据**: O'Doherty 2023 年仍被引用
- **下载链接**: https://www.judiciaryni.uk/files/judiciaryni/2024-05/King%20v%20Darren%20Gleeson.pdf
- **下载状态**: ✅ 法庭官网公开

### D5. Morrison & Enzinger 2019 "Introduction to forensic voice comparison" [P1] [OA-Preprint]
- **作者**: Morrison, G.S. & Enzinger, E.
- **出版**: in *The Routledge Handbook of Phonetics* (W.F. Katz & P.F. Assmann, Eds.)
- **下载链接 (预印本)**: https://publications.aston.ac.uk/id/eprint/38272/6/
- **下载状态**: ✅ Aston eprints 公开预印本

### D6. French 2017 "A developmental history of forensic speaker comparison in the UK" [P2] [OA-Preprint]
- **作者**: French, J.P.
- **出版**: *English Phonetics*
- **下载链接**: https://eprints.whiterose.ac.uk/id/eprint/117763/7/
- **下载状态**: ✅ White Rose eprints 公开

---

## E. Neural SV — Interpretability (我们方法的对比/参考)

### E1. PhiNet (Ma et al. 2026) [P1] [OA-Preprint]
- **作者**: Ma, Y., et al.
- **出版**: arXiv:2604.01590
- **关键数据**: nasals + vowels both crucial; learned weight + leave-one-out
- **draft 用途**: Related Work (简短提及，不重) + Discussion
- **下载状态**: ✅ 已在 `paper/PhiNet.pdf`

### E2. ExPO (Ma et al. 2025) [P2] [OA-Preprint]
- **作者**: Ma, Y., et al.
- **出版**: arXiv:2501.05729
- **关键数据**: F-ratio per-phoneme; [N-V] 非言语段高 F-ratio
- **下载链接**: https://arxiv.org/pdf/2501.05729
- **下载状态**: ✅ arXiv 公开

### E3. Thebaud et al. 2024 "Phonetic Analysis of SV through Phoneme Selection and IG" [P0] [OA]
- **出版**: Proc. Odyssey 2024
- **关键数据**: IG + phoneme alignment 首次应用于 SV pairwise
- **draft 用途**: Related Work 主要对比对象（IG 方法学）
- **下载链接**: https://www.isca-archive.org/odyssey_2024/thebaud24_odyssey.pdf
- **下载状态**: ✅ ISCA archive 公开

### E4. Wang & Li 2024 "How Phonemes Contribute to Deep Speaker Models?" [P0] [OA-Preprint]
- **作者**: Wang, M. & Li, M.
- **出版**: arXiv:2402.02730
- **关键数据**: LayerCAM + TAO 显示 vowels 优势; reliability check
- **draft 用途**: Related Work 主要对比对象（与我们结论一致的论文）
- **下载链接**: https://arxiv.org/pdf/2402.02730
- **下载状态**: ✅ arXiv 公开

### E5. Ben-Amor et al. (BA-LR) 2022/2023/2024 [P2] [OA]
- **作者**: Ben-Amor, R., et al.
- **出版**: IWBF 2022 / Interspeech 2023 / Odyssey 2024
- **下载链接**:
  - Interspeech 2023: https://www.isca-archive.org/interspeech_2023/benamor23_interspeech.pdf
  - Odyssey 2024: https://www.isca-archive.org/odyssey_2024/benamor24_odyssey.pdf
- **下载状态**: ✅ ISCA archive 公开

### E6. Zhang et al. 2023 INTERSPEECH "Visualization of Voiceprint" [P1] [已有]
- **出版**: INTERSPEECH 2023
- **下载状态**: ✅ 已在 `paper/Zhang_etal_INTERSPEECH2023_visualization_voiceprint.pdf`

### E7. Shen et al. 2025 (如果引用) [P2] [待定]
- **draft 用途**: Discussion 可能引用
- **下载状态**: 待确定具体文献

---

## F. ECAPA-TDNN 与底层方法

### F1. Desplanques, Thienpondt, Demuynck 2020 "ECAPA-TDNN" [P0] [OA-Preprint]
- **出版**: Interspeech 2020
- **DOI**: 10.21437/Interspeech.2020-2650
- **下载链接**: https://arxiv.org/pdf/2005.07143
- **下载状态**: ✅ arXiv 公开

### F2. Sundararajan, Taly, Yan 2017 "Axiomatic Attribution for Deep Networks" (Integrated Gradients 原始论文) [P0] [OA-Preprint]
- **出版**: ICML 2017
- **下载链接**: https://arxiv.org/pdf/1703.01365
- **下载状态**: ✅ arXiv 公开

### F3. Chung, Nagrani, Zisserman 2018 "VoxCeleb2: Deep Speaker Recognition" [P1] [OA-Preprint]
- **出版**: Interspeech 2018
- **下载链接**: https://arxiv.org/pdf/1806.05622
- **下载状态**: ✅ arXiv 公开

### F4. Nagrani, Chung, Zisserman 2017 "VoxCeleb: a large-scale speaker identification dataset" [P1] [OA-Preprint]
- **出版**: Interspeech 2017
- **下载链接**: https://arxiv.org/pdf/1706.08612
- **下载状态**: ✅ arXiv 公开

### F5. Baevski et al. 2020 / Xu et al. 2021 (wav2vec2 for phoneme recognition) [P2] [OA-Preprint]
- **出版**: NeurIPS 2020 / ICASSP 2021
- **下载链接**:
  - wav2vec 2.0: https://arxiv.org/pdf/2006.11477
  - wav2vec2-Phoneme: https://arxiv.org/pdf/2109.11680
- **下载状态**: ✅ arXiv 公开

---

## 下载汇总

### ✅ 可下载（17 篇）
1. ICPhS 2023 Meta-analysis (B4)
2. Smorenburg & Heeren 2021 (C1)
3. Becker et al. 2008 (C2)
4. Cavalcanti et al. 2024 (C6)
5. Kavanagh 2012 PhD (C7) — 待验证
6. Amino & Arai 2012 (C9)
7. ENFSI BPM 2021 (D1)
8. R v O'Doherty 2002 (D2)
9. R v Flynn 2008 (D3) — HTML
10. King v Gleeson 2023 (D4)
11. Morrison & Enzinger 2019 preprint (D5)
12. French 2017 preprint (D6)
13. ExPO (E2)
14. Thebaud 2024 (E3)
15. Wang & Li 2024 (E4)
16. BA-LR Interspeech 2023 (E5a)
17. BA-LR Odyssey 2024 (E5b)
18. ECAPA-TDNN (F1)
19. Integrated Gradients (F2)
20. VoxCeleb (F3, F4)
21. wav2vec2 (F5)

### ❌ 付费墙（需要二手转引或机构存档）
1. Rose 2002 书 (A1) — Smorenburg 2021 §1 已转引 p.135
2. Nolan 1983 书 (A2)
3. Jessen 2008 (A3)
4. Gold & French 2011 (B1) — 我们之前 librarian 调研已有 verbatim
5. Gold & French 2019 (B2) — 同上
6. Morrison 2016 INTERPOL (B3) — 待 forensic-voice-comparison.net 验证
7. Hughes 2017 filled pauses (C3) — 待 York eprints 验证
8. McDougall 2004 (C4) — 待 Cambridge eprints 验证
9. Morrison 2009 JASA (C5) — 待 forensic-voice-comparison.net 验证
10. Amino & Arai 2009 (C8)

### 🔍 待验证机构存档
- Morrison 2009/2016: https://forensic-voice-comparison.net/
- Hughes 2017, McDougall 2004: York eprints, Cambridge eprints
