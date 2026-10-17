# BookNLP 非 LLM 方案升级分析报告

> 更新日期：2026-10-17。研究目标是改善小说翻译前的角色提及、指代消解、说话人归属和跨章节角色聚合；不把论文 benchmark 的高分直接等同于整本小说的最佳表现。

## 1. 结论

当前最务实的路线是**先在 Python 3.12 环境把 Maverick 跑通，与 BookNLP 原有 coref 在同一文本上对比；如果跨章节角色 ID 是主要瓶颈，再单独验证 xCoRe**。

仓库已经把依赖目标改为 Python 3.12，并将 Transformers 限制在 4.x。原因是此前的 Maverick 实验在 Transformers 5.x 下遇到 DeBERTa-v3 tokenizer 兼容问题，且 sentencepiece==0.2.0 在原 Python 3.14 环境构建失败。改版本是为了降低依赖障碍，不代表模型在 MacBook Air M5 上已经实测成功。

## 2. 当前模块与合理的升级范围

| 子任务 | 当前方案 | 建议 |
|---|---|---|
| 实体/事件/超义 | BookNLP BERT 与分层标注模块 | 先保留，建立误差样例后再决定是否换底座 |
| 人名与别名聚类 | BookNLP NameCoref | 保留原结果作为重要基线；名字相似不等于同一人物 |
| 局部共指 | BookNLP LitBankCoref | Maverick 做第一个神经模型对照 |
| 跨章节 cluster merge | 需要在现有结果之上额外处理 | 若局部模型质量可接受，进一步验证 xCoRe 长文档路径 |
| 引语与说话人归属 | BookNLP quote / BERT-QA | 先按多角色对话构建专项错误集；不要因为替换共指而默认说话人也会变好 |
| 指称性别 | EM 风格推断 + 先验词表 | 暂时保留，谨慎处理名字性别先验与虚构角色 |
| 角色声线、关系和称谓规则 | 不是传统 coref 的直接输出 | 作为下游角色档案/context graph 单独建模，必要时让 LLM 复核 |

## 3. 共指后端

### 3.1 Maverick：第一优先验证

Maverick 是非 LLM 的 detect-then-cluster 共指模型。官方模型列表中包括 LitBank 专用模型 sapienzanlp/maverick-mes-litbank；官方 README 当前列出的 LitBank 分数为 78.0。它提供提及 span 与 cluster 输出，可以接入统一中间格式，再与 BookNLP 的 .entities 对齐。

安装和环境策略：

- 项目默认 Python 为 3.12；
- transformers>=4.40,<5，避免误解析 DeBERTa-v3 的 SentencePiece 模型；
- Maverick 作为可选依赖：uv sync --extra coref；
- 先用官方文档列出的 CPU 设备执行，再尝试 MPS。MPS 不是 Maverick README 明确列出的受支持设备，不能提前承诺可用或更快。

Maverick 的局部窗口质量不代表整本书的跨章节身份一致性已经解决。要在长文本中使用，仍要定义切窗、重叠区域、全局 ID 和 cluster merge 规则。

### 3.2 xCoRe：长文档阶段的候选

xCoRe 的特色是先得到窗口内 clusters，再做跨 context cluster merging；这更贴近整本小说的角色聚合问题。官方提供 input_type="long" 的接口与 LitBank/BookCoref 相关模型。它值得作为第二阶段研究对象，而不是一开始就替换全 pipeline。

兼容性注意：官方源码将 transformers==4.34 固定为依赖，与本仓库核心依赖 transformers>=4.40,<5 不兼容。因此应在单独的 Python 3.12 虚拟环境里安装和运行 xCoRe，不要把其依赖强行并入 BookNLP 主环境。

参考：[xCoRe 代码](https://github.com/SapienzaNLP/xcore)、[EMNLP 2025 论文](https://aclanthology.org/2025.emnlp-main.1737/)、[BOOKCOREF 论文](https://aclanthology.org/2025.acl-long.1197/)。

### 3.3 fastcoref、Coreferee、spacy-coref 与 GLiNER

- **fastcoref**：可作为另一条神经共指基线，但当前版本的依赖声明与主项目的 Transformers 4.x 选择不一致。用独立环境比较，避免为一个模型把主项目升级回 Transformers 5.x。
- **Coreferee**：基于 spaCy 的共指方案，适合作为不同方法的对照；不是专门的整书跨章节聚合系统。
- **spacy-coref**：ONNX 推理的轻量候选，需确认当前发行包的 Python 版本和 alpha 阶段限制；不能仅因为 ONNX 就假设更快或 M5 兼容。
- **GLiNER**：可以补充人物、称谓、组织、地点等 mention detection。本仓库提供 ner 可选依赖，但 GLiNER 不是共指模型，不能替代 Maverick/xCoRe。

## 4. 推荐架构

所有后端先转成一份标准化的 mention/cluster 中间数据，不要直接覆盖 BookNLP 的输出。至少保留：

- 原文 span、章节 ID、表面文本和 mention 类型；
- local_cluster_id 与 global_character_id 两层 ID；
- 来源引擎、模型版本、窗口和合并证据；
- 不确定状态；低置信度不自动合并。

推荐的数据流：BookNLP baseline → Maverick 局部共指 → 章节间 cluster merge → 别名/称谓规则 → 疑难候选复核 → 角色档案与 Context Graph。xCoRe 可以作为替代的长文档后端评估，而 GLiNER 只负责补充提及。

LLM 不应该直接生成全书最终 cluster。它更适合对明确候选做 merge / keep_separate / uncertain 判定，并返回原文证据。默认只自动应用高置信度结论。

## 5. POC 验收设计

### 安装与运行

1. 用 Python 3.12 重建环境：uv lock、uv sync。
2. 运行现有 881-token baseline，确认 Python 版本变更没有破坏现有输出。
3. 安装 Maverick extra：uv sync --extra coref；先 CPU 执行短样本，再跑约 5k–10k tokens、包含多角色对话的章节。
4. 把结果映射回同一份规范化原文；确认 span offset 不因文本归一化、换行处理或分窗发生漂移。
5. 再用跨章节样例测试别名、称谓、代词和角色再次出现的合并质量。只有需要整书长程聚合时，再单独启动 xCoRe 环境。

### 质量和性能

至少记录：

- 安装成功率、模型首次加载耗时与峰值内存；
- mention 覆盖率、误合并/漏合并数量；
- 跨章节角色 ID 一致性；
- CPU 与可选 MPS 的实际耗时；
- 需要人工修复的比例。

短文本的 entities 数量或 LitBank Avg.F1 不能替代小说级验证。BOOKCOREF 提供 full、split 和 gold-window 等评测方式，比较时要采用一致的评测设置。

## 6. 参考资料

1. [Maverick 官方实现与模型列表](https://github.com/SapienzaNLP/maverick-coref)
2. [Maverick PyPI 依赖信息](https://pypi.org/project/maverick-coref/)
3. [xCoRe 官方实现](https://github.com/SapienzaNLP/xcore)
4. [xCoRe 论文（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.1737/)
5. [BOOKCOREF 论文（ACL 2025）](https://aclanthology.org/2025.acl-long.1197/)
6. [BOOKCOREF 代码与数据](https://github.com/SapienzaNLP/bookcoref)
7. [fastcoref 官方仓库](https://github.com/shon-otmazgin/fastcoref)
8. [Coreferee 官方仓库](https://github.com/richardpaulhudson/coreferee)
9. [spacy-coref 包信息](https://pypi.org/project/spacy-coref/)
10. [GLiNER 官方仓库](https://github.com/urchade/GLiNER)
11. [PyTorch MPS 后端说明](https://docs.pytorch.org/docs/stable/notes/mps.html)
