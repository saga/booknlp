# BookNLP 非 LLM 方案升级分析报告

> 生成日期：2026-10-17。面向“在不使用 LLM 的前提下，BookNLP 流水线各子任务是否有更好的现代方案”。

## 1. 摘要

BookNLP 是一个面向长文档（整本书）的多任务 NLP 流水线，包含实体识别、事件/超义、引语检测与说话人归属、共指消解、指称性别推断。其核心模型基于 BERT（small 4 层 / big 6 层），发布于 2020–2021 年代。近年来非 LLM 领域在以下方向有明显进展：

- **共指消解**：Maverick（DeBERTa-large，SOTA，资源高效）与 ImCoref（进一步提升）已成为新的非 LLM SOTA；Maverick 在 LitBank 上 Avg.F1 78.4，ImCoref 79.7。
- **长文本编码**：轻量桥接模块（LBM）可增强跨段语义连续性，减少分段带来的性能损失。
- **模型底座**：DeBERTa-v3 / ModernBERT 等现代编码器普遍优于 BERT，但需要重新训练。
- **引语归属**：非 LLM 方案仍以 Muzny 两阶段 sieve 与 BookNLP 的 BERT-QA 为主；LLM 在该任务上优势明显，但非 LLM 可通过更强编码器与特征工程改进。
- **超义/事件**：基于现代编码器的序列标注可直接迁移，训练数据仍为 SemCor/LitBank。

结论：在**不使用 LLM** 的前提下，最现实的升级路径是：底座换为 DeBERTa-v3/ModernBERT，并用 Maverick/ImCoref 替换原有共指模块；其余模块复用现有架构但升级编码器与训练数据。

## 2. 当前 BookNLP 架构基线

- 实体/事件/超义：BERT + 3 层 LSTM-CRF（嵌套 NER）。
- 共指：BERT + mention-pair/链式；人名聚类 + 代词共指。
- 引语归属：BERT-QA（上下文窗口 50 token，候选打分）。
- 性别推断：IBM Model 1 风格 EM。

## 3. 共指消解：更好的非 LLM 方案

### 3.1 Maverick（推荐）

- 论文：Martinelli et al., ACL 2024, “Maverick: Efficient and Accurate Coreference Resolution Defying Recent Trends”。
- 底座：DeBERTa-large。
- 特点：detect-then-cluster；内存/速度优势（170x 更快，0.006x 训练内存）；LitBank Avg.F1 78.4。
- 非 LLM，资源友好，适合长文档。

### 3.2 ImCoref（进一步提升）

- 论文：Luo et al., 2025, “ImCoref-CeS”。
- 在 Maverick 基础上：轻量桥接模块（LBM）增强长文本编码；biaffine 评分器；混合 mention 正则化（HyMR）。
- LitBank Avg.F1 79.7（纯监督，非 LLM）。
- 可选 LLM Checker-Splitter 提升至 81.8，但非 LLM 版本仍优于 Maverick。

### 3.3 其它非 LLM 候选

- FastCoref / LingMess：速度/准确率较好，但 LitBank 上略逊于 Maverick。
- Longformer-based（longdoc/Dual-cache）：长文本友好，但性能略低。

## 4. 实体识别与嵌套 NER

- 现代方案：基于 DeBERTa-v3 的嵌套 NER（span-based 或 layered CRF）。
- 数据：LitBank + OntoNotes；BookNLP 已有 968K token 训练数据。
- 建议：保留 layered NER 架构，底座升级为 DeBERTa-v3-large/ModernBERT；重新训练。

## 5. 引语说话人归属

- 非 LLM SOTA 仍以 Muzny 两阶段 sieve（确定性规则）与 BookNLP BERT-QA 为主。
- LLM（Llama-3）在该任务上显著优于非 LLM，但用户要求非 LLM。
- 建议：升级底座为 DeBERTa-v3，扩展上下文窗口，加入角色别名/性别先验特征；或混合 sieve 规则 + 神经打分。

## 6. 超义与事件

- 底座升级即可；训练数据 SemCor/LitBank。
- 可考虑多任务联合训练（实体+超义+事件）以共享表示。

## 7. 指称性别推断

- 现有 EM 方案仍合理；可加入现代代词集合与更强先验。
- 无需 LLM。

## 8. 推荐的非 LLM 升级路线

1. 共指模块替换为 Maverick/ImCoref（DeBERTa-large），保留 BookNLP 的人名聚类与性别推断。
2. 全局底座从 BERT 升级为 DeBERTa-v3-large 或 ModernBERT。
3. 长文本编码引入 LBM（轻量桥接）。
4. 引语归属升级底座 + 特征工程。
5. 训练/推理启用 torch.compile、混合精度、批处理优化。

## 9. 参考文献

- Martinelli et al., ACL 2024, Maverick.
- Luo et al., 2025, ImCoref-CeS (arXiv 2510.10241).
- Muzny et al., 2017, Two-stage sieve quote attribution.
- Bamman et al., LitBank coreference dataset.

## 10. 附：性能对比（LitBank Avg.F1）

| 方法 | Avg.F1 | 底座 |
|---|---|---|
| BookNLP (BERT) | ~76-79 | BERT |
| Maverick | 78.4 | DeBERTa-large |
| ImCoref | 79.7 | DeBERTa-large |
| ImCoref-CeS (LLM) | 81.8 | DeBERTa + LLM |
```

Done.<|end|><|start|> assistants to=write<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="path">
/Users/saga/code-repos/booknlp/docs/research_report_non_llm.md