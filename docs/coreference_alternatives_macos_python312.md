# BookNLP 共指消解方案：Python 3.12 / Apple Silicon

> 更新日期：2026-10-17  
> 目标环境：saga/booknlp、Python 3.12、MacBook Air M5 16 GB  
> 目标：优先降低依赖安装障碍，并找到可用于小说翻译预处理的角色提及、局部共指与跨章节聚合方案。

## 结论

**切换到 Python 3.12 后，Maverick 值得重新尝试。** 官方 maverick-coref 包声明支持 Python 3.8+，而此前失败的两个主要问题分别是 Python 3.14 下构建 sentencepiece==0.2.0，以及 Transformers 5.x 对 DeBERTa-v3 SentencePiece tokenizer 的兼容问题。当前仓库已经改为 Python 3.12，并把 Transformers 限制到 4.x；Maverick 作为可选依赖，不强迫所有使用者安装。

本次仅完成仓库元数据和文档改造，**没有在 MacBook Air M5 上实际创建环境或运行模型**。所以“更容易安装”有官方依赖声明依据，但是否能完整运行、MPS 是否可用、推理速度和小说质量仍待 POC 验证。

推荐顺序：

1. 在 Python 3.12 主环境安装 BookNLP，确认现有 baseline 没有回归。
2. 安装 uv sync --extra coref，先用 Maverick CPU 跑短样本，再跑章节样本。
3. 如果主要问题是跨章节身份合并，再隔离 Python 3.12 环境验证 xCoRe。
4. 可选加测 fastcoref / Coreferee；GLiNER 只能作为提及抽取补充，而不是共指替代。

## 1. 为什么 Python 3.12 更适合这次实验

当前仓库使用 Python 3.14，锁文件也曾在 requires-python >=3.14 条件下生成，依赖选择包括 Transformers 5.x。这个组合增加了模型代码与 tokenizer 的兼容风险。

现在项目设定为：

- requires-python = ">=3.12,<3.13"
- torch>=2.2,<3
- transformers>=4.40,<5
- spacy>=3.7,<4
- Maverick optional extra：maverick-coref==1.0.7

Maverick 官方 setup.py 声明 transformers>=4.34、sentencepiece==0.2.0 和 Python 3.8+。因此 Python 3.12 与 Transformers 4.x 满足其发布元数据范围；这并不排除特定依赖组合仍可能出现运行时问题。

## 2. 备选方案

| 方案 | Python 3.12 安装路线 | 长篇适配 | 建议 |
|---|---|---|---|
| **Maverick** | 主环境可选 extra；用 Transformers 4.x，避免 5.x | 主要解决窗口内共指；全书级处理仍需窗口与全局 ID 策略 | **第一优先** |
| **xCoRe** | 独立环境，官方依赖固定 Transformers 4.34，与主项目当前 >=4.40 冲突 | 目标就是 cross-context merging 与 long-document coreference | 第二阶段 |
| **fastcoref** | 单独环境。当前版本依赖 Python <3.14，且较新版本走 Transformers 5.x，和主环境约束不兼容 | 可以对照局部共指质量；跨窗口需要额外聚合 | 可选对照 |
| **Coreferee** | 可在 Python 3.12 环境评估 spaCy 依赖链 | 不是专门的整本书跨章节合并方案 | 可选轻量对照 |
| **spacy-coref / ONNX** | Python 3.12 可作为候选，但需验证当前发行包限制与依赖 | 轻量推理的潜力，不等于长篇质量更好 | 暂后 |
| **GLiNER** | 主项目 ner extra | 负责发现人物/称谓等提及，不聚合共指链 | 辅助工具 |
| **BookNLP 原 coref + 规则** | 默认依赖，无需另装模型 | 作为现有 baseline；规则可补充书内别名与称谓 | 保留基线 |

来源：[Maverick 官方仓库](https://github.com/SapienzaNLP/maverick-coref)、[Maverick PyPI](https://pypi.org/project/maverick-coref/)、[xCoRe](https://github.com/SapienzaNLP/xcore)、[fastcoref](https://github.com/shon-otmazgin/fastcoref)、[Coreferee](https://github.com/richardpaulhudson/coreferee)、[spacy-coref](https://pypi.org/project/spacy-coref/)、[GLiNER](https://github.com/urchade/GLiNER)。

### 2.1 Maverick：最小化变更的 POC

在 Python 3.12 下，先安装项目默认依赖并验证 baseline：

~~~bash
uv python install 3.12
uv sync
uv run python -m spacy download en_core_web_sm
~~~

再安装 Maverick：

~~~bash
uv sync --extra coref
~~~

对短文本的最小推理示例（使用官方接口，CPU 作为 M5 初始基线）：

~~~python
from maverick import Maverick

model = Maverick(
    hf_name_or_path="sapienzanlp/maverick-mes-litbank",
    device="cpu",
)
prediction = model.predict("Elizabeth Bennet entered the room. She looked around.")
print(prediction)
~~~

注意：

- 第一次运行可能需要从 Hugging Face 下载模型权重；应单独记录下载与模型加载时间。
- Maverick 官方 README 列出的设备是 CPU/CUDA，没有把 Apple MPS 声明为受支持后端。先用 CPU 确认输出，之后再试 MPS；不能提前宣称 MPS 一定可运行或更快。
- LitBank 模型与 OntoNotes 模型训练的 singleton 和标注习惯不同，小说场景建议先试 sapienzanlp/maverick-mes-litbank，同时检查它的输出是否符合你的应用。
- 不要把 Maverick cluster 的编号直接覆盖 BookNLP 的 coreference ID；需要通过 span 对齐和统一的中间层进行映射。

### 2.2 xCoRe：整本小说的跨章节聚合

xCoRe 的价值在于先做窗口内共指，再通过跨 context cluster merging 处理长文档。官方接口支持以 input_type="long" 处理长文本，并要求依据当前设备内存设置 max_length。[官方仓库](https://github.com/SapienzaNLP/xcore)。

但它的源码固定 transformers==4.34，与本仓库当前 transformers>=4.40,<5 冲突。因此不要在主 BookNLP 环境运行下面的安装命令；应建立单独的 Python 3.12 环境：

~~~bash
uv venv --python 3.12 .venv-xcore
source .venv-xcore/bin/activate
python -m pip install --upgrade pip
python -m pip install xcore-coref==0.1.3
~~~

这只是起始安装路线，具体 wheel 与依赖解析必须实测。先 CPU 跑短文本，再测试长文档参数；MPS 支持不能从 PyTorch 的平台能力直接推定。

### 2.3 fastcoref 与其它候选

fastcoref 适合作为另一种窗口内神经共指算法来比较，但当前版本的依赖线与主项目不同，因此应独立安装，不要为了它把主环境的 Transformers 改回 5.x。Coreferee 可以作为基于 spaCy 的对照；spacy-coref 的 ONNX 路线有轻量推理价值，但须核对当前发行版的 Python 声明。GLiNER 只适合补足提及抽取。

## 3. 统一接口，避免模型相互污染

模型输出统一映射成中间记录，例如：

~~~json
{
  "book_id": "novel-001",
  "chapter_id": "ch-012",
  "start_char": 1832,
  "end_char": 1845,
  "surface": "the captain",
  "mention_type": "nominal",
  "local_cluster_id": "maverick:ch-012:cluster-7",
  "global_character_id": null,
  "source_engine": "maverick",
  "model_revision": null,
  "evidence": []
}
~~~

关键约束：

- span offset 必须映射到一份固定的规范化原文；切窗和标准化前后不能混用 offsets。
- 保留 local cluster 与 global character 两层 ID，跨章节合并可以撤销。
- 记录模型、权重 revision、窗口范围与 merge 证据。
- 低置信度的两个 cluster 默认不自动合并。
- 角色档案（声线、关系、称谓、事件）是下游语义整理任务，不是传统 coref 模型自动给出的可靠事实。

建议流程：BookNLP 基线 → Maverick 局部共指 → 候选 cluster 对齐 → 章节间归并 → 疑难项复核 → 角色档案/Context Graph。需要时用本地 LLM 对有限候选作 merge / keep_separate / uncertain 判断，并要求输出原文证据。

## 4. POC 验收

### 安装和兼容性

- uv lock 在 Python 3.12 约束下完成，不依赖旧的 Python 3.14 锁文件。
- uv sync 与 uv sync --extra coref 均完成。
- 现有短文本 baseline 能重复生成同样格式的输出。
- Maverick CPU 推理成功，模型权重可缓存和再次加载。
- 单独记录 Python、PyTorch、Transformers、SentencePiece 和模型 revision。

### 质量

用同一份规范化文本对比 BookNLP 与 Maverick，首轮样例至少包含：

- 别名/昵称/敬称；
- 同一个称谓对应不同角色；
- 多人对话中的 she/he/they；
- the captain、her father 等描述性 mention；
- 隔多个章节再次出现的人物。

不要只比较 cluster 个数。记录错误合并、漏合并、跨章节 ID 一致性、人工纠正数量与推理时间。只有需要 long-range cluster merge 时，再加入 xCoRe 对照。

## 5. 实施建议

**第一步只做 Python 3.12 + Transformers 4.x + Maverick。** 这条路线改动最小，适合确认此前的失败是否主要来自 Python 3.14 构建与 Transformers 5.x 兼容问题。

如果 Maverick 在 M5 上能运行但跨章节一致性仍不足，再投入 xCoRe。不要一开始就把 Maverick、xCoRe、fastcoref 一股脑塞进同一个依赖环境；它们的 Transformers 约束并不一致。

## 参考

- [Maverick 官方 README 与模型](https://github.com/SapienzaNLP/maverick-coref)
- [Maverick PyPI 包](https://pypi.org/project/maverick-coref/)
- [xCoRe 官方实现](https://github.com/SapienzaNLP/xcore)
- [xCoRe 论文](https://aclanthology.org/2025.emnlp-main.1737/)
- [BOOKCOREF 论文](https://aclanthology.org/2025.acl-long.1197/)
- [fastcoref](https://github.com/shon-otmazgin/fastcoref)
- [Coreferee](https://github.com/richardpaulhudson/coreferee)
- [spacy-coref / ONNX](https://pypi.org/project/spacy-coref/)
- [GLiNER](https://github.com/urchade/GLiNER)
- [PyTorch MPS 文档](https://docs.pytorch.org/docs/stable/notes/mps.html)
