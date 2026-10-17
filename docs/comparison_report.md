# BookNLP Baseline vs Maverick（非 LLM）对比报告

## 1. 实验设置
- 输入：`booknlp/data/english/pride.short.txt`（881 tokens）
- Baseline：BookNLP small pipeline（entity/quote/supersense/event/coref）
- New：Maverick（DeBERTa-large，非 LLM SOTA 共指），Python 3.12 环境

## 2. 结果
| 指标 | BookNLP | Maverick |
|---|---|---|
| mention 数 | 174 | 160 |
| 簇/ID 数 | 50 | 16 |
| 平均簇大小 | 3.5 | 10.0 |

## 3. 分析
- **mention 检测相近**：两者 mention 数接近（174 vs 160），说明 Maverick 的 mention 检测与 BookNLP 实体识别在同一量级，差异不大。
- **聚类策略差异显著**：Maverick 仅产生 16 个簇，BookNLP 产生 50 个 ID。Maverick 的 detect-then-cluster 更激进，倾向于把更多 mention 合并，减少碎片化；BookNLP 通过人名聚类 + 代词共指，保留更多细粒度实体。
- **风险**：Maverick 的过度合并可能把不同实体错误合并（降低精度），而 BookNLP 的保守策略可能产生过多碎片（降低召回）。仅从数量无法判断孰优。

## 4. 结论
- **数量对比不能证明提升**：需要 gold 标注的共指数据（如 LitBank）计算 MUC/B3/CEAF F1 才能判断。
- **Maverick 可行但需适配**：在 Python 3.12 + transformers 4.x 下可运行，但与 BookNLP 的实体/引语/性别模块需对齐（mention span、coref ID 映射）。
- **建议**：在 LitBank 测试集上做端到端评估；若 Maverick F1 更高，替换共指模块；否则保留 BookNLP 并借鉴其 DeBERTa 底座与长文本编码改进。