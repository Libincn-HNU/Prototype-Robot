import pyltp
from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser
from snownlp import SnowNLP
import numpy as np
import pandas as pd
import re,os,jieba, codecs
from itertools import chain
import pycorrector

LTP_DATA_DIR = '/Users/sunhongchao/Documents/craft/09_Dialogue/resources/ltp_data_v3.4.0'  # ltp模型目录的路径
RESOURCE_DIR = ''

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词，
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标，
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 句法分析
stopwords_path = os.path.join(RESOURCE_DIR, 'stopwords.txt')
abbreviation_path = os.path.join(RESOURCE_DIR, 'abbreviations.txt')

segmentor = Segmentor()  
segmentor.load(cws_model_path)  
postagger = Postagger()  
postagger.load(pos_model_path)  
recognizer = NamedEntityRecognizer()  
recognizer.load(ner_model_path)  
parser = Parser() 
parser.load(par_model_path)
stopwords = codecs.open(stopwors_path, encoding='utf-8', 'r').readlines()
abbreviations = codecs.open(abbreviation_path, encoding='utf-8', 'r').readlines()


def sentence_rewrite(input_sentence:str):

    tmp_str = ''
    s = SnowNLP(input_sentence)

    # 纠错 : 基于 pycorrector
    tmp_str , _ = pycorrector.correct(input_sentence)

    # 专有名字/缩略词扩充 : 结合 词典 README.md 中有词典相关资源
    pass

    # 去除非中文
    input_str = re.sub(r'[^\u4e00-\u9fa5]+','', input_sentence)

    # 繁简转化
    han_list = s.han(input_str)

    # 关键词
    keywords_list = s.keywords(5)
    print('key words list', keywords_list)

    # 拼音
    pinin_list = s.pinyin 
         
def lexical_analysis(input_sentence:str):

    # 分词 
    words = segmentor.segment(input_str) 
    words_list = list(words)   
    print('word list', words_list)

    # 去停用词
    words_list_after_stopwords = []
    for word in words_list:
        if word not in stopwords:
            words_list_after_stopwords.append(word)

    # 词性标注
    postags = postagger.postag(words)  
    postags_list = list(postags)  
    print('pos list', postags_list)

    # 命名实体识别
    nertags = recognizer.recognize(words, postags)  
    nertags_list = list(nertags)  
    print('ner list', nertags_list)

    return words_list, words_list_after_stopwords, postags, nertags_list

def syntax_analysis(input_sentence:str):
    # 调用词法分析的结果
    words_list, words_list_after_stopwords, postags, nertags_list, _, _ = lexical_analysis(input_sentence)
    # 句法结构分析
    pass

    # 依存句法分析
    arcs = parser.parse(words, postags)  
    print('parser list', '\t'.join('%d: %s' %(arc.head, arc.relation) for arc in arcs))

    return arcs

def semantic_analysis(input_sentence:str):
    # 词义消歧

    # 语义角色标注

def discourse_analysis(input_document:str):

    # 长句压缩/消除冗余/文本摘要

    # 分句
    if input_str.strip():
        sentences_list = []
        line_split = re.split(r'[。！；？]',input_str.strip())
        line_split = [line.strip() for line in line_split if line.strip() not in ['。','！','？','；'] and len(line.strip())>1]
        sentences_list.append(line_split)
        return sentences_list

    # 指代消解 : 使用段落上文信息进行指代消解 零指代/
    pass

def document_analysis():
    """结构化的文档转化为一个个段落
    """

def merge_all():
    pass

if __name__ == "__main__":
    query_deal('我在北京chaoyang工作，做的是自然语言处理')
    # query_deal('德清在哪里')


# 语义表示的要不要加 