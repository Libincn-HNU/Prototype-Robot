import sys
import pickle
from elasticsearch import RequestsHttpConnection, Elasticsearch
from elasticsearch.helpers import bulk


all_index_name = 'du_reader_index_test'
all_index_type = 'new_detail'
all_file_name = 'corpus-step-1.pkl'

class Search(object):
    
    def __init__(self, ip='127.0.0.1'):
        self.es = Elasticsearch([ip], port=9200)

    def create_index(self, index_name=all_index_name, index_type=all_index_type):
        """
        创建索引
        """
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
        if self.es.indices.exists(index=index_name) is not True:
            res = self.es.indices.create(index=index_name, body=_index_mappings)
            print('create new index ')
            print(res)



class ElasticObj(object):
    
    def __init__(self, index_name,index_type, ip ="127.0.0.1"):
        self.index_name =index_name
        self.index_type = index_type
        self.es = Elasticsearch([ip],http_auth=('elastic', 'password'),port=9200)

    def Get_Data_By_Body(self, inputs):
        doc = {"query": {"bool": {'should':[{'match':{'query':inputs}}]}}}
        # doc = {"query":{"match_phrase":{"answer":inputs}}}

        _searched = self.es.search(index=self.index_name, body=doc, size=20)

        answer_list = []
        idx = 0
        for hit in _searched['hits']['hits']:
            answer_list.append(hit['_source']['answer'])
            # print( ' es-score ' + str(hit['_score']) + ' idx '+ str(idx) + ' ### ' +  'query is ', hit['_source']['query'], ' ###  answer is ', hit['_source']['answer'], ' ### ', len(hit['_source']['answer']))
            idx = idx + 1

        return(answer_list)

    def Get_Data_By_Answer(self, inputs):
        doc = {"query":{"match_phrase":{"answer":inputs}}}

        _searched = self.es.search(index=self.index_name, body=doc, size=20)

        answer_list = []
        for hit in _searched['hits']['hits']:
            answer_list.append(hit['_source']['answer'])
            print('query is ', hit['_source']['query'], ' ###  answer is ', hit['_source']['answer'], ' ### ', len(hit['_source']['answer']))

        return(answer_list)

    def Delete_Data_By_Body(self, inputs):
        doc = {"query":{"match_phrase":{"answer":inputs}}}

        result = self.es.delete_by_query(index=self.index_name, body=doc)
        print(result)

    def bulk_Index_Data(self, input_list):
        '''
        用bulk将批量数据存储到es
        :return:
        '''

        if len(input_list) is  0:
            print("input list is none, use demo list")
            input_list = [
                {'query':'你好','answer':'你好啊'},
                {'query':'hello', 'answer':'hi'},
                {'query':'天气不错啊','answer':'是的啊，情况万里，天气很好'},
                {'query':'心情不错啊','answer':'是的，心情特别好'},
                {'query':'心情好','answer':'今天真高兴'},
                {'query':'今天开心','answer':'今天真高兴'}]

        ACTIONS=[]
        i = 1

        for line in input_list:
            action  = {
                '_index':self.index_name,
                '_type':self.index_type,
                '_id':i, 
                '_source':{
                    'query':line['query'],
                    'answer':line['answer']
                }
            }
            i += 1

            if i % 100000 == 0:
                print('index is ', i)
            ACTIONS.append(action)

        success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)


def build():
    build = Search()
    build.create_index()

    input_list = []
    count = 0

    import json
    with open('/Users/apollo/Documents/Bot/corpus/mrc/dureader/preprocessed/merge/search.merge.json', mode='r', encoding='utf-8') as f:
        dataset = f.readlines()

    for item in dataset:
        item = json.loads(item)

        if item['answer_spans'] is None: # 没有答案
            continue
        if item['question_type'] is not 'DESCRIPTION': # 不是描述型
            continue

        for document in item['documents']:
            if document['is_selected'] is True:
                # print('*'*10)
                # print(document['segmented_title'])
                # print('question', item['question'])
                # print('title', document['title'])
                # print('&'*10)
                # print(document['segmented_paragraphs'][document['most_related_para']])
                # print('most related paragraph', document['paragraphs'][document['most_related_para']])
                input_list.append({'query':item['question'], document['paragraphs'][document['most_related_para']]})

                break

    obj = ElasticObj('brc_test_name', 'brc_test_type')
    obj.bulk_Index_Data(input_list)
    obj.Get_Data_By_Body(sys.argv[1])

def search():
    build = Search()
    build.create_index()

    input_list = []

    obj = ElasticObj('new_qa_name', 'new_qa_type')
    obj.Get_Data_By_Body(sys.argv[1])


if __name__ == '__main__':
    #build()

    search()
    
#class Operate(object):
#    "使用 指定的IP， 端口， index_name 进行 查询"
#
#    def __init__(self, ip='127.0.0.1'):
#        self.es = Elasticsearch([ip], port=9200)
#
#
#    def search_query(self):
#        obj = ElasticObj(all_index_name, all_index_type)
#        answer_list = obj.Get_Data_By_Body(sys.argv[1])
#        print(answer_list)
#
#    def search_query_and_delete(self):
#        obj = ElasticObj(all_index_name, all_index_type)
#        answer_list = obj.Get_Data_By_Body(sys.argv[1])
#        obj.Delete_Data_By_Body(sys.argv[1])
#    
#    def search_answer_and_delete(self):
#        obj = ElasticObj(all_index_name, all_index_type)
#        answer_list = obj.Get_Data_By_Answer(sys.argv[1])
#        obj.Delete_Data_By_Body(sys.argv[1])
