import sys
from elasticsearch import RequestsHttpConnection, Elasticsearch
from elasticsearch.helpers import bulk

class Search(object):

    def __init__(self, ip='127.0.0.1'):
        self.es = Elasticsearch([ip], port=9200)

    def create_index(self, index_name='chat_corspu_1', index_type='ott_date'):
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
        print("Performed %d actions " % success)

    def Get_Data_By_Body(self, inputs):
        doc = {"query": {"bool": {'should':[{'match':{'query':inputs}}]}}}
        
        _searched = self.es.search(index=self.index_name, body=doc)

        print('_searched', _searched)
        for hit in _searched['hits']['hits']:
            print('query is : ', hit['_source']['query'], ' answer is :', hit['_source']['answer'])

if __name__ == '__main__':
    search = Search()
    search.create_index()

    obj = ElasticObj('qa_info', 'qa_detail')
    
    tmp_count = 0

    if False:
        input_list = []

        query_list, answer_list = [], []
        with open('../../corpus/dialogue/new_corpus.txt', mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            print('all corpus length is :', len(lines))
            for idx in range(0, len(lines) - 3, 3):

                if lines[idx].startswith('M') and lines[idx+1].startswith("M") and lines[idx+2].startswith('E'):

                    tmp_query = lines[idx][2:].strip()
                    tmp_answer = lines[idx+1][2:].strip()
                    if len(tmp_query) is 0 or len(tmp_answer) is 0:
                        print('tmp_query is : ', tmp_query)
                        print('tmp_answer is :', tmp_answer)
                        
                        print(idx)
                        tmp_count += 1 
                    else:
                        query_list.append(tmp_query)
                        answer_list.append(tmp_answer)
                else:
                    print('error in ', idx)
                    break

        for tmp_query, tmp_answer in zip(query_list, answer_list):
            input_list.append({'query':tmp_query,'answer':tmp_answer})

        print("all tmp count is ", tmp_count)
        print("build input list done")


        obj.bulk_Index_Data(input_list)
    obj.Get_Data_By_Body(sys.argv[1])
