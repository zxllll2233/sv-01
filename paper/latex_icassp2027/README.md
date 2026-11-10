# ICASSP 2027 Submission

从 `paper/latex_v7_en/`（APSIPA 2025 版）迁移过来，改用 ICASSP 标准 spconf 双栏格式。

## 文件

| 文件 | 用途 |
|---|---|
| `main.tex` | 主稿（spconf 双栏格式） |
| `refs.bib` | BibTeX 引用（IEEEbib.bst 兼容） |
| `spconf.sty` | ICASSP/ICIP 官方样式文件（源自 ICASSP 2026 Paper Kit） |
| `IEEEbib.bst` | IEEE 参考文献样式 |
| `figures/fig1_shape_2x2.pdf` | Figure 1: shape-normalized intervention curves |
| `figures/fig2_paired_ig_evidence.pdf` | Figure 2: paired IG sign-resolved saliency（双栏跨列） |

## 编译方法

### 本地

```bash
cd paper/latex_icassp2027
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**注意用 `bibtex` 而不是 `biber`**（IEEEbib 是 BibTeX 而非 biblatex）。

### Overleaf

1. 新建 project，上传本目录所有文件（保留 `figures/` 子目录结构）
2. Menu → Compiler → **pdfLaTeX**
3. Compile

## 与 APSIPA 版的主要差异

| 项 | APSIPA v7 | ICASSP 2027 |
|---|---|---|
| Document class | `\documentclass[conference,a4paper]{APSIPA2025}` | `\documentclass{article}` + `\usepackage{spconf}` |
| 页面 | A4（geometry 手工设定） | Letter（spconf 内置） |
| 作者块 | `\authorblockN` + `\authorblockA` | `\name` + `\address`（spconf 原生） |
| 参考文献 | biblatex-ieee + biber | IEEEbib.bst + bibtex |
| 页眉 | fancyhdr APSIPA 标识 | spconf 默认（无页眉） |
| 页数上限 | 8 页 | **5 页**（4 页正文 + 1 页 references-only） |

## 页数控制

ICASSP 上限 **5 页**：
- 4 页可含正文 / 图 / 表 / references
- 第 5 页只能放 references / funding acknowledgement / Compliance with Ethical Standards

当前稿子较长（Intro + Related + Framework + Experiments + Discussion + Conclusion），编译后**很可能超 5 页**，需要压缩。压缩优先级建议：
1. 合并 Related Work 三个 subsection → 单段
2. Discussion 缩到半段
3. Prior track cross-track consistency 用一段带过
4. Table `\resizebox` 已是最紧格式，不能再压
5. Figure 1 可以从双栏跨列（figure*）转成单栏（figure）
6. 若第 5 页有正文，触发 desk reject——用 `\balance` 或调段落密度

## 待补内容

同 APSIPA 版：
1. **作者信息**：`\name` / `\address` 里的 Author One/Two/Three 占位符
2. **Kendall $\tau$ 值**：`[to be filled]` 待 cross-track 后处理脚本产出

## 关键 ICASSP 2027 时间节点

- Full Paper Submission Deadline: **2026-09-16**
- Acceptance Notification: 2027-01-13
- Camera Ready: 2027-01-27

## 来源

- spconf.sty / IEEEbib.bst / Template.tex 骨架 → ICASSP 2026 官方 Paper Kit（`cmsworkshops.com/ICASSP2026/papers/PaperFormat/`）
  - ICASSP 2027 官网目前尚未上线独立 kit，但历届均沿用同一套 spconf/IEEEbib，仅需把标题里的 "ICASSP 2027" 就绪即可
- 正文 / 图片 / bib → `paper/latex_v7_en/`（APSIPA 2025 版）
