# -*- coding: utf-8 -*-
"""
BookNLP 顶层入口模块。

本模块提供两个对外接口：
1. `BookNLP` 类：供 Python 代码调用（`from booknlp.booknlp import BookNLP`）。
2. `proc()`：命令行入口，支持 `-l/-i/-o/--id` 参数。

整体流程：
- 根据 language 创建对应的 BookNLP 实现（目前仅支持英文 EnglishBookNLP）。
- 调用 process(inputFile, outputFolder, idd) 启动整条 NLP 流水线。
"""

import sys
import argparse
# 屏蔽 transformers 的冗长日志，避免干扰用户输出
from transformers import logging
logging.set_verbosity_error()
from pathlib import Path
import os
from booknlp.english.english_booknlp import EnglishBookNLP


class BookNLP():
    """BookNLP 顶层封装类，负责根据语言选择具体实现。"""

    def __init__(self, language, model_params):
        # 目前仅支持英文；后续可扩展其它语言
        if language == "en":
            self.booknlp = EnglishBookNLP(model_params)

    def process(self, inputFile, outputFolder, idd):
        """运行完整流水线：inputFile 输入文本，outputFolder 输出目录，idd 为文件名前缀。"""
        self.booknlp.process(inputFile, outputFolder, idd)


def proc():
    """命令行入口函数：解析参数并运行 BookNLP。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', help='Currently on {en}', required=True)
    parser.add_argument('-i', '--inputFile', help='Filename to run BookNLP on', required=True)
    parser.add_argument('-o', '--outputFolder', help='Folder to write results to', required=True)
    parser.add_argument('--id', help='ID of text (for creating filenames within output folder)', required=True)

    args = vars(parser.parse_args())

    language = args["language"]
    inputFile = args["inputFile"]
    outputFolder = args["outputFolder"]
    idd = args["id"]

    print("tagging %s" % inputFile)

    # 校验语言合法性
    valid_languages = set(["en"])
    if language not in valid_languages:
        print("%s not recognized; supported languages: %s" % (language, valid_languages))
        sys.exit(1)

    # 默认使用全流水线 + small 模型
    model_params = {
        "pipeline": "entity,quote,supersense,event,coref", "model": "small",
    }

    booknlp = BookNLP(language, model_params)
    booknlp.process(inputFile, outputFolder, idd)


# 直接运行本文件时启动 CLI
if __name__ == "__main__":
    proc()