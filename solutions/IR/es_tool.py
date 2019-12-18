import sys
from elasticsearch import RequestsHttpConnection, Elasticsearch
from elasticsearch.helpers import bulk

class Search(object):

    def __init__(self, ip='127.0.0.1'):
        self.es = Elasticsearch([ip], port=9200)

    def create_index(self, index_name='new', index_type='ott_date'):
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
            print(res)

class ElasticObj(object):
    
    def __init__(self, index_name,index_type, ip ="127.0.0.1"):
        '''
        :param index_name: 索引名称
        :param index_type: 索引类型
        '''
        self.index_name =index_name
        self.index_type = index_type
        self.es = Elasticsearch([ip],http_auth=('elastic', 'password'),port=9200)

    def bulk_Index_Data(self, input_list):
        '''
        用bulk将批量数据存储到es
        :return:
        '''

        if len(input_list) is not 0:
            print("input list is none, use demo list")
            input_list = [
                {'query':'你好','answer':'你好啊'},
                {'query':'hello', 'answer':'hi'},
                {'query':'天气不错啊','answer':'是的啊，情况万里，天气很好'},
                {'query':'心情不错啊','answer':'AAA'},
                {'query':'心情好','answer':'BBB'},
                {'query':'今天开心','answer':'CCC'}]

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
        print("Performed %d actions " % success)

    def Get_Data_By_Body(self, inputs):
        doc = {"query": {"bool": {'should':[{'match':{'query':inputs}}]}}}
        
        _searched = self.es.search(index=self.index_name, body=doc)

        print('_searched', _searched)
        for hit in _searched['hits']['hits']:
            print('key is : ', hit['_source']['query'], ' query is :', hit['_source']['answer'])

if __name__ == '__main__':
    search = Search()
    search.create_index()
    
    obj = ElasticObj('qa_info', 'qa_detail')
    obj.bulk_Index_Data()
    obj.Get_Data_By_Body(sys.argv[1])
