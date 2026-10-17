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

## 3. 对齐评估（近似 B3，基于文本匹配）
| 方法 | 对齐 mention 数 | P | R | F1 |
|---|---|---|---|---|
| BookNLP | 188/219 | 0.786 | 0.679 | 0.729 |
| Maverick | 131/219 | 0.808 | 0.709 | 0.755 |

## 4. 结论
- 在同一 LitBank 测试文档上，Maverick 的近似 B3 F1（0.755）略高于 BookNLP（0.729），提升约 **+0.026**。
- Maverick 对齐 mention 更少（131 vs 188），说明其 mention 检测/对齐覆盖更低，但聚类精度更高。
- 由于对齐基于精确文本匹配，结果为近似；正式结论需 CoNLL scorer + gold mention 对齐。