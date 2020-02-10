import sys
import pickle
from elasticsearch import RequestsHttpConnection, Elasticsearch
from elasticsearch.helpers import bulk


class Build(object):
    def __init__(self, ip='127.0.0.1'):
        self.es = Elasticsearch([ip], port=9200)


    def create_index(self, index_name='chat_corpus_2', index_type='ott_date', file_name):
        _index_mappings = {
            "mappings":{
                "properties":{
                    "query":{
                        'type':'text',
                        },
                    'answer':{
                        'type':'text',
                        }
                    }
                }
            }
        
        if self.es.indices.exists(index=index_name) is not True: # index 不存在 则创建
            res = self.es.indices.create(index=index_name, body=_index_mappings)
            print(res)

            with open(file_name, 'rb') as f:
                results = pickle.load(f)
                for item in results:
                    idx = 0
                    while idx < len(item) - 1:
                        self.es.index(index=index_name, body={"query":item[idx] ,"answer":item[idx+1]})

class Operate(object):
    "使用 指定的IP， 端口， index_name 进行 查询"

    def __init__(self, ip='127.0.0.1'):
        self.es = Elasticsearch([ip], port=9200)

    def create_index(self, index_name='chat_corpus_1', index_type='ott_date'):
        _index_mappings = {
            "mappings":{
                "properties":{
                    "query":{
                        'type':'text',
                        },
                    'answer':{
                        'type':'text',
                        }
                    }
                }
            }
        if self.es.indices.exists(index=index_name) is not True: # index 不存在 则创建
            res = self.es.indices.create(index=index_name, body=_index_mappings)
            print(res)

    def search_query(self):
        self.create_index()
        obj = ElasticObj('qa_info', 'qa_detail')
        answer_list = obj.Get_Data_By_Body(sys.argv[1])
        print(answer_list)

    def search_query_and_delete(self):
        self.create_index()
        obj = ElasticObj('qa_info', 'qa_detail')
        answer_list = obj.Get_Data_By_Body(sys.argv[1])
        obj.Delete_Data_By_Body(sys.argv[1])
    
    def search_answer_and_delete(self):
        self.create_index()
        obj = ElasticObj('qa_info', 'qa_detail')
        answer_list = obj.Get_Data_By_Answer(sys.argv[1])
        obj.Delete_Data_By_Body(sys.argv[1])

class ElasticObj(object):
    
    def __init__(self, index_name,index_type, ip ="127.0.0.1"):
        self.index_name =index_name
        self.index_type = index_type
        self.es = Elasticsearch([ip],http_auth=('elastic', 'password'),port=9200)

    def Get_Data_By_Body(self, inputs):
        doc = {"query": {"bool": {'should':[{'match':{'query':inputs}}]}}}
        # doc = {"query":{"match_phrase":{"answer":inputs}}}

        _searched = self.es.search(index=self.index_name, body=doc, size=10)

        answer_list = []
        idx = 0
        for hit in _searched['hits']['hits']:
            answer_list.append(hit['_source']['answer'])
            print( ' es-score ' + str(hit['_score']) + ' idx '+ str(idx) + ' ### ' +  'query is ', hit['_source']['query'], ' ###  answer is ', hit['_source']['answer'], ' ### ', len(hit['_source']['answer']))
            idx = idx + 1

        return(answer_list)

    def Get_Data_By_Answer(self, inputs):
        doc = {"query":{"match_phrase":{"answer":inputs}}}

        _searched = self.es.search(index=self.index_name, body=doc, size=10)

        answer_list = []
        for hit in _searched['hits']['hits']:
            answer_list.append(hit['_source']['answer'])
            print('query is ', hit['_source']['query'], ' ###  answer is ', hit['_source']['answer'], ' ### ', len(hit['_source']['answer']))

        return(answer_list)

    def Delete_Data_By_Body(self, inputs):
        doc = {"query":{"match_phrase":{"answer":inputs}}}

        result = self.es.delete_by_query(index=self.index_name, body=doc)
        print(result)


def build(index_name = 'chat_corpus_2', file_name='corpus-step-1.pkl'):
    build = Build()
    build.create_index(index_name=index_name, file_name= file_name)

def search():
    ope = Operate()
    ope.create_index('chat_corpus_2')


if __name__ == '__main__':
    build()
    