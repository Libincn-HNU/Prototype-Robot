import sys
import pickle
from elasticsearch import RequestsHttpConnection, Elasticsearch
from elasticsearch.helpers import bulk


all_index_name = 'brc_index_name'
all_index_type = 'brc_index_type'

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
            print( ' es-score ' + str(hit['_score']) + ' idx '+ str(idx) + ' ### ' +  'query is ', hit['_source']['query'], ' ###  answer is ', hit['_source']['answer'], ' ### ', len(hit['_source']['answer']))
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

            if i % 10000 == 0:
                print('index is ', i)
            ACTIONS.append(action)

        success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)

def build():
    build = Search()
    build.create_index()

    input_list = []
    count = 0

    import json
    dataset = []
    with open('search.merge.json', mode='r', encoding='utf-8') as f:
        dataset = f.readlines()

    for item in dataset:
        item = json.loads(item)

        if 'answer_spans' not in item.keys():
            continue

        for document in item['documents']:
            if document['is_selected'] is True:
                input_list.append({'query':item['question'], 'answer':document['paragraphs'][document['most_related_para']]})
                break

    print('input list length', len(input_list))

    obj = ElasticObj('brc_test_name', 'brc_test_type')
    obj.bulk_Index_Data(input_list)
    obj.Get_Data_By_Body(sys.argv[1])

def search():
    #build = Search()
    #build.create_index()
    #print('begin search')

    obj = ElasticObj('brc_test_name', 'brc_test_type')
    obj.Get_Data_By_Body(sys.argv[1])


if __name__ == '__main__':
    # build()
    search()
