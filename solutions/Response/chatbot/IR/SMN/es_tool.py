import sys
import pickle
from elasticsearch import RequestsHttpConnection, Elasticsearch
from elasticsearch.helpers import bulk


all_index_name = 'new_info_3'
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
    with open(all_file_name, 'rb') as f:
        results = pickle.load(f)
        for item in results:
            idx = 0
            while idx < len(item) -1:
                input_list.append({'query':item[idx], 'answer': item[idx+1]})
                count = count + 1

                if count % 100000 == 0:
                    print({'query':item[idx], 'answer': item[idx+1]})

                idx = idx + 1


    obj = ElasticObj('new_qa_name', 'new_qa_type')
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
