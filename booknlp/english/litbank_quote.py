# -*- coding: utf-8 -*-
"""
引语处理模块：QuoteTagger + QuotationAttribution。

- QuoteTagger：基于引号符号检测引语边界（", ', — 等），标记 token.inQuote。
- QuotationAttribution：对每个引语，用 BERT 从上下文候选（实体/其它引语）中选择说话人。

对 NLP 小白的解释：
- 引语（quote）：直接引语，如 "Yes," she said。
- 说话人归属（attribution）：判断这句引语是谁说的。
"""
import re
from collections import Counter

class QuoteTagger:
	

	def tag(self, toks):

		predictions=[]
		currentQuote=[]
		curStartTok=None
		lastPar=None

		quote_symbols=Counter()

		for tok in toks:
			if tok.text == "“" or tok.text == "”" or tok.text == "\"" or tok.text == "“":
				quote_symbols["DOUBLE_QUOTE"]+=1
			elif tok.text == "‘" or tok.text == "’" or tok.text == "'":
				quote_symbols["SINGLE_QUOTE"]+=1
			elif tok.text == "—":
				quote_symbols["DASH"]+=1


		quote_symbol="DOUBLE_QUOTE"
		if len(quote_symbols) > 0:
			quote_symbol=quote_symbols.most_common()[0][0]

		for tok in toks:

			w=tok.text

			for w_idx, w_char in enumerate(w):
				if w_char== "“" or w_char == "”" or w_char == "\"":
					w="DOUBLE_QUOTE"
				elif w_char == "‘" or w_char == "’" or w_char == "'":
					if w_idx == 0:
						suff=w[w_idx+1:]
						if suff != "s" and suff != "d" and suff != "ll" and suff != "ve":
							w="SINGLE_QUOTE"

			# start over at each new paragraph
			if tok.paragraph_id != lastPar and lastPar is not None:

				if len(currentQuote) > 0:
					predictions.append((curStartTok, tok.token_id-1))
				curStartTok=None
				currentQuote=[]

			if w == quote_symbol:

				if curStartTok is not None:

					if len(currentQuote) > 0:
						predictions.append((curStartTok, tok.token_id))
						currentQuote.append(tok.text)

					curStartTok=None
					currentQuote=[]
				else:
					curStartTok=tok.token_id

			
			if curStartTok is not None:
				currentQuote.append(tok.text)

			lastPar=tok.paragraph_id

		for start, end in predictions:
			for i in range(start, end+1):
				toks[i].inQuote=True

		return predictions



