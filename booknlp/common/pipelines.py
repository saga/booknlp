# -*- coding: utf-8 -*-
"""
通用数据结构与 Spacy 流水线模块。

本模块定义了 BookNLP 中最基础的两类对象：
- `Entity`：一个实体 mention（span），记录其在句子/文档中的位置、类型、所属引语等。
- `Token`：一个词 token，记录其文本、POS、依存、字节偏移等。

以及 `SpacyPipeline`：封装 spaCy，把原始文本转换为 BookNLP 的 Token 列表（分词、POS、依存解析、分句、分段）。

对 NLP 小白的解释：
- Token（词元）：把句子切成的最小单位，如 "Tom"、"was"。
- POS（词性）：名词/动词/形容词等。
- 依存关系（dependency）：词与词之间的语法依存，如 "nsubj"（主语）。
- Entity（实体）：文本中指称某事物的片段，如人名、地名。
"""

import re

class Entity:
    """表示一个实体 mention（一个文本片段）。"""

    def __init__(self, start, end, entity_id=None, quote_id=None, quote_eid=None, proper=None, ner_cat=None, in_quote=None, text=None):
        # start/end：在当前句子中的词索引（ inclusive ）
        self.start = start
        self.end = end
        self.entity_id = entity_id          # 共指 ID（后续填充）
        self.quote_id = quote_id            # 所属引语 ID
        self.quote_eid = quote_eid          # 引语中的实体 ID
        self.proper = proper                # PROP / NOM / PRON
        self.ner_cat = ner_cat              # PER / FAC / GPE / ...
        self.in_quote = in_quote            # 是否在引语内
        self.text = text                    # 实体文本
        self.quote_mention = None           # 对应的引语 mention
        self.global_start = None            # 在全文 token 序列中的起始
        self.global_end = None              # 在全文 token 序列中的结束

    def __str__(self):
        return ("%s %s %s %s %s %s %s %s" % (self.global_start, self.global_end, self.entity_id, self.proper, self.ner_cat, self.in_quote, self.quote_eid, self.text))


class Token:
    """表示一个词 token，包含其文本、位置、POS、依存、事件标注等。"""

    def __init__(self, paragraph_id, sentence_id, index_within_sentence_idx, token_id, text, pos, fine_pos, lemma, deprel, dephead, ner, startByte):
        self.text = text
        self.paragraph_id = paragraph_id        # 段落 ID
        self.sentence_id = sentence_id          # 句子 ID
        self.index_within_sentence_idx = index_within_sentence_idx  # 句内词序
        self.token_id = token_id                # 全文词 ID
        self.lemma = lemma                      # 词元（lemma）
        self.pos = pos                          # 粗粒度 POS（NOUN/VERB/...）
        self.fine_pos = fine_pos                # 细粒度 POS（NN/VBZ/...）
        self.deprel = deprel                    # 依存关系
        self.dephead = dephead                  # 依存父节点词 ID
        self.ner = ner                          # NER 标签（本流水线不使用 spaCy NER）
        self.startByte = startByte              # 在原文中的字节起始
        self.endByte = startByte + len(text)   # 字节结束
        self.inQuote = False                    # 是否在引语内
        self.event = "O"                        # 事件标注

    def __str__(self):
        # 输出为 TSV 一行，便于写入 .tokens 文件
        return '\t'.join([str(x) for x in [self.paragraph_id, self.sentence_id, self.index_within_sentence_idx, self.token_id, self.text, self.lemma, self.startByte, self.endByte, self.pos, self.fine_pos, self.deprel, self.dephead, self.event]])

    @classmethod
    def convert(self, sents):
        """把句子列表转换成 Token 列表（仅用于简单场景/测试）。"""
        toks = []
        i = 0
        cur = 0
        for sidx, sent in enumerate(sents):
            for widx, word in enumerate(sent):
                token = Token(0, sidx, widx, i, word, None, None, None, None, None, None, cur)
                toks.append(token)
                i += 1
                cur += len(word) + 1
        return toks

    @classmethod
    def deconvert(self, toks):
        """把 Token 列表按 sentence_id 还原成句子列表。"""
        sents = []
        sent = []
        lastSid = None
        for tok in toks:
            if lastSid is not None and tok.sentence_id != lastSid:
                sents.append(sent)
                sent = []
            sent.append(tok)
            lastSid = tok.sentence_id

        if len(sent) > 0:
            sents.append(sent)

        return sents


from spacy.tokens import Doc

class SpacyPipeline:
    """封装 spaCy，提供文本 -> Token 列表的转换。"""

    def __init__(self, spacy_nlp):
        self.spacy_nlp = spacy_nlp
        # 允许处理超长文本（整本书）
        self.spacy_nlp.max_length = 10000000

    def filter_ws(self, text):
        """把空格/换行/制表符转成占位符，避免 TSV 输出破坏列结构。"""
        text = re.sub(" ", "S", text)
        text = re.sub(r"[\n\r]", "N", text)
        text = re.sub(r"\t", "T", text)
        return text

    def tag_pretokenized(self, toks, sents, spaces):
        """对已分词的文本打标签（较少使用）。"""
        doc = Doc(self.spacy_nlp.vocab, words=toks, spaces=spaces)
        for idx, token in enumerate(doc):
            token.sent_start = sents[idx]

        for name, proc in self.spacy_nlp.pipeline:
            doc = proc(doc)

        return self.process_doc(doc)

    def tag(self, text):
        """主入口：输入原始文本，返回 Token 列表。"""
        doc = self.spacy_nlp(text)
        return self.process_doc(doc)

    def process_doc(self, doc):
        """把 spaCy Doc 转换为 BookNLP Token 列表，同时记录段落/句子边界。"""
        tokens = []
        skipped_global = 0
        paragraph_id = 0
        current_whitespace = ""
        sentence_id = 0
        for sid, sent in enumerate(doc.sents):
            skipped_in_sentence = 0
            skips_in_sentence = []
            curSkips = 0
            for w_idx, tok in enumerate(sent):
                if tok.is_space:
                    curSkips += 1
                skips_in_sentence.append(curSkips)

            hasWord = False

            for w_idx, tok in enumerate(sent):
                if tok.is_space:
                    skipped_global += 1
                    skipped_in_sentence += 1
                    current_whitespace += tok.text
                else:
                    # 遇到空行，段落号 +1
                    if re.search(r"\n\n", current_whitespace) is not None:
                        paragraph_id += 1

                    hasWord = True

                    # 计算依存父节点在当前句子中的索引（考虑空格跳过的偏移）
                    head_in_sentence = tok.head.i - sent.start
                    skips_between_token_and_head = skips_in_sentence[head_in_sentence] - skips_in_sentence[w_idx]
                    token = Token(paragraph_id, sentence_id, w_idx - skipped_in_sentence, tok.i - skipped_global, self.filter_ws(tok.text), tok.pos_, tok.tag_, tok.lemma_, tok.dep_, tok.head.i - skipped_global - skips_between_token_and_head, None, tok.idx)
                    tokens.append(token)
                    current_whitespace = ""

            if hasWord:
                sentence_id += 1

        return tokens

