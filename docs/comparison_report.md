# BookNLP Baseline vs 新方案对比报告

## 1. Baseline（当前代码）

运行命令：
```
python -c "from booknlp.booknlp import BookNLP; ..."
```
输入：`booknlp/data/english/pride.short.txt`（881 tokens）  
输出目录：`docs/baseline/`

Baseline 统计：
- tokens：881
- entities（mention）：174
- 唯一共指 ID 数：50
- 引语：28
- 角色（characters）：18

## 2. 新方案运行状态

尝试使用 Maverick（DeBERTa-large，非 LLM SOTA 共指）作为新方案，但受限于当前环境：
- `maverick-coref` 依赖 `sentencepiece 0.2.0`，在 Python 3.14 下构建失败；
- 即使 `--no-deps` 安装，运行时需要 `hydra/scipy/pandas/nltk/tiktoken` 等额外依赖；
- 最终因 transformers 5.x 与 DeBERTa-v3 的 `spm.model`（tiktoken 解析）不兼容而无法完成推理。

因此，本报告**无法提供在同一输入上的新方案端到端输出**，仅能基于文献报告的指标进行对比。

## 3. 文献指标对比（LitBank，Avg.F1）

| 方法 | Avg.F1 | 底座 | 是否 LLM |
|---|---|---|---|
| BookNLP (BERT) | ~76-79 | BERT | 否 |
| Maverick | 78.4 | DeBERTa-large | 否 |
| ImCoref | 79.7 | DeBERTa-large | 否 |
| ImCoref-CeS (LLM) | 81.8 | DeBERTa + LLM | 是 |

## 4. 结论

- 在非 LLM 条件下，Maverick/ImCoref 相比 BookNLP 的 BERT 共指有约 **+2~+4 点** Avg.F1 提升（LitBank）。
- 要在本仓库中实际跑出新方案数据，需要：降低 Python 到 3.10/3.11、使用 transformers 4.x、并完成 Maverick 与 BookNLP 流水线的适配（mention/coref ID 对齐）。
- 建议后续在兼容环境中完成端到端对比实验。

## 5. 后续方案分析

由于 Maverick 在当前 Python 3.14 / Apple Silicon 环境下无法完成端到端运行，替代模型、整本小说的跨章节共指、隔离 Python 环境与本地 LLM 补充策略，另见 [macOS / Python 3.14 共指消解替代方案分析](coreference_alternatives_macos_python314.md)。该报告区分了官方依赖声明与本机实测结果；替代方案的 M5 性能仍需 POC 验证。
