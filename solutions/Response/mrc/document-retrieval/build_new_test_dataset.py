import json
import jieba

from es_tool_new import *

output_list = []

obj = ElasticObj("brc_test_name", "brc_test_type")

# documents ["paragraphs":[], "segmented_graph":[], "title", "segment_title",
# "question", "question_type", "fact_or_opinion", "question_id"]

with open("new_test_question.txt", mode="r", encoding="utf-8") as f:
    for idx, item in enumerate(f.readlines()):
        query = item.strip()
        segmented_question = list(jieba.cut(query))
        question_type = "DESCRIPTION"
        fact_or_opinion = "FACT"
        question_id = idx

        answers = obj.Get_Data_By_Body(query)

        documents = []

        for answer in answers:
            documents.append({"paragraohs" : [answer], 
                              "segmented_paragraphs":[ list(jieba.cut(answer))],
                              "tile" : "标题",
                              "segmented_tile":["标题"]})
        
        
        merge_info = {"question":query,
                  "segmented_question":segmented_question,
                  "question_id":question_id,
                  "question_type":question_type,
                  "fact_or_opinion":fact_or_opinion,
                  "documents": documents
                 }
        print("&"*100)
        print(merge_info)

        output_list.append(merge_info)

import codecs

with open("build.test.json", mode="w", encoding="utf-8") as f:
    for item in output_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
