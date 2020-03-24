import jieba
import pyltp
from snownlp import SnowNLP
from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser
import os

# Paramenters
LTP_DATA_DIR = '/Users/sunhongchao/Documents/craft/09_Dialogue/resources/ltp_data_v3.4.0'  # ltp模型目录的路径

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 分词模型路径， 模型名称为'parser.model'

segmentor = Segmentor()  
segmentor.load(cws_model_path)  
postagger = Postagger()  
postagger.load(pos_model_path)  
recognizer = NamedEntityRecognizer()  
recognizer.load(ner_model_path)  
parser = Parser() 
parser.load(par_model_path)

def query_deal(input_str:str):

    # 1. 问句改写 
    # 1.1 纠错
    # pycorrector
    # import pycorrector
    # corrected_sent, detail = pycorrector.correct('少先队员应该为老人让坐')

    # 1.2 指代消解
    # todo

    # 1.3 专有名字/缩略词扩充

    # 1.4 省略消解

    # 1.5 文本归一化

    # 1.6 复述生成

    # 1.7 繁简转化
    # snownlp

    # 2. 长句压缩/消除冗余/文本摘要

    # 3. 分词
    words = segmentor.segment(input_str) 
    words_list = list(words)   
    print('word list', words_list)
    # 4. 词性标注
    postags = postagger.postag(words)  
    postags_list = list(postags)  
    print('pos list', postags_list)
    # 5. 命名实体识别
    netags = recognizer.recognize(words, postags)  
    netags_list = list(netags)  
    print('ner list', netags_list)
    # 6. 句法分析
    arcs = parser.parse(words, postags)  
    print('parser list', '\t'.join('%d: %s' %(arc.head, arc.relation) for arc in arcs))
    # 7. 关键词
    s = SnowNLP(input_str)
    keywords_list = s.keywords(5)
    print('key words list', keywords_list)

    # 8. 意图识别

    # 9. 情感分析
    # SnowNLP

    # 10. 拼音
    # snownlp

    # 11. QQ 匹配
    # 字面匹配
    # todo:语义匹配

    # 12. QA  匹配
    # 字面匹配
    # todo:语义匹配

    # 13. 语义索引

    # 14. 知识库匹配
    # todo : 构造知识库

    
    # other
    # # 信息量衡量
         

    return words_list, postags_list, netags_list


if __name__ == "__main__":
    query_deal('我在北京chaoyang工作，做的是自然语言处理')
    # query_deal('德清在哪里')