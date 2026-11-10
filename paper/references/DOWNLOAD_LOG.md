# Download Log — paper/references/pdfs/

> 最后更新: 2026-07-23
> 目的: 记录当前 ICASSP 2027 稿 (`paper/latex_icassp2027/main.tex`) 实际引用文献的 PDF 下载状态
> 命名规范: `年份-期刊简称-论文名.pdf`

## 背景（2026-07-23 重下）

旧 pdfs/ 目录里 27 个文件全是 **0 字节空占位符**（git 治理时内容丢失），且大量对应**已废弃的 forensic phonetics framing**（Amino_Arai / BA-LR / Gold_PhD / Kavanagh / King_v_Gleeson / Morrison 等），当前稿根本不引用。已全部删除，按当前 `refs.bib` 实际引用重新下载。

## ✅ 已成功下载并验证 (18 篇)

对每篇均用 `pypdf` 验证：能正常打开 + 页数合理 + 首页文本与标题吻合（非错误页伪装）。

| # | 文件名 | 页数 | 来源 |
|:--:|---|:--:|---|
| 1 | `2017-ICML-Axiomatic-Attribution.pdf` | 11 | arXiv 1703.01365 |
| 2 | `2018-BMVC-RISE.pdf` | 17 | arXiv 1806.07421 |
| 3 | `2019-EURASIP-Phonetic-Info-Speaker-Embedding.pdf` | 17 | SpringerOpen 官方 (10.1186/s13636-019-0166-8) |
| 4 | `2020-ACL-ERASER.pdf` | 16 | arXiv 1911.03429 |
| 5 | `2020-CVPRW-Score-CAM.pdf` | 11 | arXiv 1910.01279 |
| 6 | `2020-Interspeech-ECAPA-TDNN.pdf` | 5 | arXiv 2005.07143 |
| 7 | `2021-CVPR-Guided-IG.pdf` | 13 | arXiv 2106.09788 |
| 8 | `2021-Interspeech-wav2vec2-Phoneme.pdf` | 8 | arXiv 2109.11680 |
| 9 | `2021-TIP-LayerCAM.pdf` | 14 | mmcheng.net 官方 (TIP 2021) |
| 10 | `2022-ICLR-Axiomatic-Explanations-Visual-Search.pdf` | 26 | arXiv 2103.00370 (ICLR 2022 正式版) |
| 11 | `2022-Interspeech-Reliable-Visualization-SV.pdf` | 5 | ISCA archive (li22l) |
| 12 | `2023-Interspeech-Visualizing-Data-Augmentation-SV.pdf` | 5 | ISCA archive (li23p) |
| 13 | `2023-Interspeech-Voiceprint-Feature-Visualization.pdf` | 5 | ISCA archive (zhang23x) |
| 14 | `2024-ICASSPW-How-Phonemes-Contribute.pdf` | 6 | arXiv 2402.02730 |
| 15 | `2024-Odyssey-Phonetic-Analysis-SV-IG.pdf` | 8 | ISCA archive (thebaud24) |
| 16 | `2025-APSIPA-Multilevel-Adversarial-Training-SV.pdf` | 6 | APSIPA 2025 proceedings 官方 |
| 17 | `2025-SPL-ExPO.pdf` | 5 | arXiv 2501.05729 |
| 18 | `2025-TASLP-PhiNet.pdf` | 14 | arXiv 2604.01590 |

## ❌ 被引用但无免费 PDF (3 篇 — IEEE 付费墙，无 arXiv/镜像)

librarian 通过 arXiv API + Semantic Scholar + DBLP + 机构库确认这 3 篇**全网无免费版**，仅 IEEE Xplore 付费。需 IEEE 权限或作者/馆际索取。

| bib key | 论文 | venue | DOI |
|---|---|---|---|
| `song2024multilingual` | Introducing Multilingual Phonetic Information to Speaker Embedding for SV | ICASSP 2024, pp.10091-10095 | 10.1109/ICASSP48485.2024.10446546 |
| `he2024multiview` | Multi-View Speaker Embedding Learning for Enhanced Stability and Discriminability | ICASSP 2024, pp.10081-10085 | 10.1109/ICASSP48485.2024.10448494 |
| `fang2024improving` | Improving SV With Noise-Aware Label Ensembling and Sample Selection | IEEE/ACM TASLP vol.32 2024, pp.2988-3001 | 10.1109/TASLP.2024.3407527 |

> 注：`song2024multilingual` 的**代码**开源在 github.com/zds-potato/multilingual-phonetic-sv，但无论文 PDF。

## ⏭️ 未下载 (1 篇 — 书籍)

| bib key | 说明 |
|---|---|
| `pearl2009causality` | Judea Pearl《Causality: Models, Reasoning, and Inference》，专著，无单篇 PDF，不下载 |

## 引用勘误（下载过程中发现）

1. **`zhang2023visualizing`**（已修 refs.bib）：原作者错写 `Jing Zhang and Lantian Li and Dong Wang`（Jing Zhang 虚构 + 漏 Askar Hamdulla）。经 ISCA archive + DBLP 核实，正确为 **Pengqi Li, Lantian Li, Askar Hamdulla, Dong Wang**，pp.2243-2247，DOI 10.21437/Interspeech.2023-1298。已修正 refs.bib 作者/页码/DOI。
   - ⚠️ 连带：正文 §2 / §3.2 仍写 "Zhang et al.~\cite{zhang2023visualizing}"，作者变 Pengqi Li 后应改 "Li et al."（待用户定，因与 li2024phonemes/li2022reliable 的 Li et al. 相邻易混）。

2. **`hamilton2021model`**（✅ 已修 refs.bib）：原标题 "Model-Agnostic Explainability for Visual Search" + 作者 "Ryan Hamilton, Scott Zhang, Steve Hartwell, Antonio Torralba" **几乎全错**。经核实真实论文为 **"Axiomatic Explanations for Visual Search, Retrieval, and Similarity Learning"**（ICLR 2022），作者 **Mark Hamilton, Scott Lundberg, Lei Zhang, Stephanie Fu, William T. Freeman**，arXiv 2103.00370。已改为 @inproceedings（ICLR 2022），修正标题+作者+venue。PDF 已下载（2022-ICLR-Axiomatic-Explanations-Visual-Search.pdf）。

3. **`fan2020cnceleb`**（已删）：refs.bib 中存在但 main.tex 未引用的死条目，已从 refs.bib 移除。

4. **`song2024multilingual` / `he2024multiview` / `fang2024improving`**（✅ 二次核实，bib 字段全部正确，仅 PDF 付费墙）：三篇均 IEEE Xplore / TASLP 付费墙，经 Semantic Scholar API（`isOpenAccess:false`, `CLOSED`）+ arXiv（0 结果）+ DBLP（`access:closed`）+ 作者主页 + GitHub 多源交叉确认，**全网无免费版**。bib 里的 title/author/pages/DOI 与权威源逐字符一致，无需修正。fang2024 的 Google Scholar cluster 里有个 drive.google.com PDF，但不可验证/权限受限，未采用。→ 需你用 IEEE 权限或馆际索取。

5. **`zhang2023visualization`**（✅ 二次核实，完全正确，非 `zhang2023visualizing`，是另一篇）："A Study on Visualization of Voiceprint Feature"（Interspeech 2023, pp.2233-2237, DOI 10.21437/Interspeech.2023-1286），作者 Jian Zhang, Liang He, Xiaochen Guo, Jing Ma。ISCA archive + DBLP + researchr + Semantic Scholar 四源一致，无需修正。PDF 已下载（2023-Interspeech-Voiceprint-Feature-Visualization.pdf）。
