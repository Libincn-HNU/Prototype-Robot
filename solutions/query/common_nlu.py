import pyltp
import pycorrector
import numpy as np
import pandas as pd
import re, os, jieba, codecs

from pyltp import Segmenter, Postagger, NamedEntityRecognizer, Parser
from snownlp import SnowNLP
from itertools import chain

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词，
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标，
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 句法分析
stopwords_path = os.path.join(RESOURCE_DIR, 'stopwords.txt')
abbreviation_path = os.path.join(RESOURCE_DIR, 'abbreviations.txt')

segmenter = Segmenter()  
segmenter.load(cws_model_path)  
postagger = Postagger()  
postagger.load(pos_model_path)  
recognizer = NamedEntityRecognizer()  
recognizer.load(ner_model_path)  
parser = Parser() 
parser.load(par_model_path)
stopwords = codecs.open(stopwors_path, encoding='utf-8', 'r').readlines()
abbreviations = codecs.open(abbreviation_path, encoding='utf-8', 'r').readlines()


def text_rewrite(input_text:str)->str:
    """
    句子改写相关实现
    """

    tmp_str = input_text
    s = SnowNLP(tmp_str)

    # 纠错 : 基于 pycorrector
    tmp_str , _ = pycorrector.correct(tmp_str)

    # 专有名字/缩略词扩充 : 结合 词典 README.md 中有词典相关资源
    pass

    # 去除非中文
    tmp_str = re.sub(r'[^\u4e00-\u9fa5]+','', tmp_str)

    # 繁简转化
    tmp_str = s.han(tmp_str)

    # # 关键词
    # keywords_list = s.keywords(5)
    # print('key words list', keywords_list)

    # # 拼音
    # pinin_list = s.pinyin 

    return tmp_str
         
def lexical_analysis(input_text:str):
    """
    词法分析相关实现
    """

    tmp_str = input_text

    # 分词 
    words = segmenter.segment(tmp_str) 
    word_list = list(words)   

    # 词性标注
    pos_tags = postagger.postag(words)  
    pos_list = list(pos_tags)  

    # 命名实体识别
    ner_tags = recognizer.recognize(words, pos_tags)  
    ner_list = list(ner_tags)  

    # # 去停用词
    # words_after, pos_after, ner_after = [], [], []
    # for word in words_list:
    #     if word not in stopwords:
    #         words_after.append(word)

    return word_list, pos_list, ner_list

def syntax_analysis(input_text:str):

    tmp_str = input_text

    # 调用词法分析的结果
    words_list, pos_list, ner_list = lexical_analysis(tmp_str)
    # 句法结构分析
    pass

    # 依存句法分析
    arcs = parser.parse(words, postags)  
    print('parser list', '\t'.join('%d: %s' %(arc.head, arc.relation) for arc in arcs))

    return arcs

def discourse_analysis(input_document:str):

    # 长句压缩/消除冗余/文本摘要
    # TextRank
    # 1. 找语料 2. 找其他方法

    # 分句
    if input_str.strip():
        sentences_list = []
        line_split = re.split(r'[。！；？]',input_str.strip())
        line_split = [line.strip() for line in line_split if line.strip() not in ['。','！','？','；'] and len(line.strip())>1]
        sentences_list.append(line_split)
        return sentences_list

    # 指代消解 : 使用段落上文信息进行指代消解 零指代
    # 
    pass

def semantic_analysis(input_sentence:str):
    # 词义消歧
    # 词-释义 表
    # 1. 探索语料 有语料 有监督 2. 少量语料半监督 3. 无语料 无监督
    # bert 可以解决一词多义的问题吗？

    # 语义角色标注
    # ltp