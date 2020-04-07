import pyltp
# import pycorrector
import numpy as np
import pandas as pd
import re, os, jieba, codecs

from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser, SentenceSplitter
from snownlp import SnowNLP
from itertools import chain

LTP_DATA_DIR = '/Users/sunhongchao/Documents/craft/09_Dialogue/resources/ltp_data_v3.4.0'
RESOURCE_DIR = '/Users/sunhongchao/Documents/craft/09_Dialogue/resources/'

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词，
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标，
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 句法分析
stopwords_path = os.path.join(RESOURCE_DIR, 'stopwords.txt')
abbreviation_path = os.path.join(RESOURCE_DIR, 'abbreviations.txt')

segmenter = Segmentor()  
segmenter.load(cws_model_path)  
postagger = Postagger()  
postagger.load(pos_model_path)  
recognizer = NamedEntityRecognizer()  
recognizer.load(ner_model_path)  
parser = Parser() 
parser.load(par_model_path)
stopwords = codecs.open(stopwords_path, encoding='utf-8', mode='r').readlines()
abbreviations = codecs.open(abbreviation_path, encoding='utf-8', mode='r').readlines()

stopwords = [item.strip() for item in stopwords]
abbreviations = [item.strip() for item in abbreviations]


def text_rewrite(input_text:str)->str:
    """
    句子改写相关实现
    """

    tmp_str = input_text
    s = SnowNLP(tmp_str)

    # # 纠错 : 基于 pycorrector
    # correct_str , _ = pycorrector.correct(tmp_str)

    # 专有名字/缩略词扩充 : 结合 词典 README.md 中有词典相关资源
    pass

    # 去除非中文
    just_chinese_str = re.sub(r'[^\u4e00-\u9fa5]+','', tmp_str)

    # 繁简转化
    han_str = s.han

    # 关键词
    keywords_list = s.keywords(5)

    # 拼音
    pinin_list = s.pinyin 

    return '', '', just_chinese_str, han_str, keywords_list, pinin_list 


def discourse_analysis(input_text:str):
    """
    # 长句压缩/消除冗余/文本摘要
    #   TextRank 
    #       1. 找语料 2. 找其他方法
    #   snonlp
    # 指代消解
    #   探索方法(Transformer)
    """
    # 分句
    sentences = list(SentenceSplitter.split(input_text))

    # 摘要
    s = SnowNLP(input_text)
    summary_list = s.summary(limit=3)

    # 指代消解 : 使用段落上文信息进行指代消解 零指代
    # 
    pass

    return sentences, summary_list, ''

def lexical_analysis(input_text:str):
    """
    词法分析相关实现
    """
    # 分词 
    words = segmenter.segment(input_text) 
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
    """
    # 句法结构分析
    # 依存句法分析
    """

    words, pos, ner = lexical_analysis(input_text)

    # 句法结构分析
    pass

    # 依存句法分析
    arcs = parser.parse(words, pos)  
    print('parser list', '\t'.join('%d: %s' %(arc.head, arc.relation) for arc in arcs))

    return arcs


def semantic_analysis(input_sentence:str):
    # 词义消歧
    # 词-释义 表
    # 1. 探索语料 有语料 有监督 2. 少量语料半监督 3. 无语料 无监督
    # bert 可以解决一词多义的问题吗？

    # 语义角色标注
    # ltp

    pass