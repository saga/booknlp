# BookNLP Baseline vs Maverick（非 LLM）对比报告

> 更新日期：2026-10-17。本文保留仓库现有短文本对比统计，并把 Python 3.12 设为正式实验环境。

## 1. 实验设置

- 输入：booknlp/data/english/pride.short.txt（881 tokens）
- Baseline：BookNLP small pipeline（entity/quote/supersense/event/coref）
- New：Maverick（DeBERTa-large，非 LLM 共指），Python 3.12 环境
- Baseline 输出：docs/baseline/

## 2. 当前记录的结果

| 指标 | BookNLP | Maverick |
|---|---:|---:|
| mention 数 | 174 | 160 |
| 簇/ID 数 | 50 | 16 |
| 平均 mention/簇 | 3.5 | 10.0 |

这里的簇数和平均簇大小来自模型输出的描述性统计，并非 gold-label 评估结果。不同系统的 mention detection 机制不同，因此两侧 mention 数也不能直接当成模型召回率比较。

## 3. 分析

- **mention 数量接近**：160 与 174 的差异不大，但这不能证明两种系统找到了相同的提及；需按原文 span 逐项对齐。
- **聚类策略差异显著**：Maverick 仅产生 16 个簇，BookNLP 产生 50 个 ID。Maverick 的 detect-then-cluster 在这段输入上形成更大的 cluster；这既可能减少碎片，也可能把不同实体错误合并。
- **不能从簇数判断准确性**：BookNLP 较多的 ID 可能是细粒度区分，也可能是漏合并；Maverick 较少的 ID 可能是成功归并，也可能是过度合并。
- **必须核验 span 对齐**：模型的 cluster ID 没有跨模型语义；只有把 mention spans 映射到同一份规范化原文后，才能比较对应关系。

## 4. Python 3.12 依赖策略

仓库的项目声明现在固定 Python 3.12，并将 Transformers 限制在 4.x：

- Python：>=3.12,<3.13
- Transformers：>=4.40,<5
- Maverick optional dependency：maverick-coref==1.0.7

安装步骤见 README 与 [Python 3.12 共指替代方案分析](coreference_alternatives_macos_python312.md)。之前由 Python 3.14 解析生成的 uv.lock 已不适用于此环境，需要在可访问包索引的机器上重新生成。

## 5. 下一步评估

1. 确认 174 / 160 个 mention 都能映射回统一原文字符 span。
2. 抽查差异，尤其是角色别名、敬称、普通名词和代词。
3. 在人工标注的 LitBank/小说样例上计算 MUC、B³、CEAF 或 CoNLL F1，同时报告误合并与漏合并。
4. 对跨章节一致性单独评估；Maverick 的局部共指结果并不自动解决整本书的 global character ID 问题。
5. 在 MacBook Air M5 上记录模型加载耗时、CPU 推理耗时和峰值内存。Maverick 官方 README 列出 CPU/CUDA，但未明确承诺 MPS 支持，因此 MPS 仍需独立验证。

## 6. 指标与来源说明

Maverick 官方模型列表当前列出的 LitBank 分数为 78.0；此前文档中出现的 78.4 可能来自另一模型/评测版本。ImCoref 的 79.7 也只有在数据集、切分、singleton 规则和评测脚本一致时才可比较。论文指标是外部参考，不是本仓库上述 881-token 输入上的实测分数。

参考：[Maverick 官方项目](https://github.com/SapienzaNLP/maverick-coref)、[Maverick PyPI](https://pypi.org/project/maverick-coref/)、[BOOKCOREF 论文](https://aclanthology.org/2025.acl-long.1197/)。
