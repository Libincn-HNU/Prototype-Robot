# -*- coding:utf-8 -*-
import jieba
import json
import codecs
import flask

from flask import (Flask,render_template,url_for, jsonify, send_from_directory, request, flash)
from flask_cors import cross_origin, CORS
import datetime, time, json, codecs, os, xlrd
from werkzeug.utils import secure_filename
from es_jd.es_tool_jd import ElasticObj

app = Flask(__name__)
CORS(app)
app.secret_key='service in inter-credit'

@app.route('/')
def index():
    return("welcome to MRC test " + "*" * 100 )

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def run():
    input_text = request.args.get("inputText")
    input_text = '你好'
    obj = ElasticObj("brc_index_name", "brc_index_type")
    query = input_text.strip()
    segmented_question = list(jieba.cut(query))
    question_type = "DESCRIPTION"
    fact_or_opinion = "FACT"
    question_id = 0
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

    output_file = "/export/home/sunhongchao1/Prototype-Robot/corpus/mrc/dureader/preprocessed/testset/build.test.json" 
    model_folder = "/export/home/sunhongchao1/Prototype-Robot/services/models/model_mrc/bidaf"
    vocab_dir = "/export/home/sunhongchao1/Prototype-Robot/solutions/response/mrc/document-retrieval/data/"
    with open(output_file, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(merge_info, ensure_ascii=False) + "\n")

    import os
    str=('python /export/home/sunhongchao1/Prototype-Robot/solutions/response/mrc/document-retrieval/run.py --predict --algo BIDAF --vocab_dir ' + vocab_dir + ' --model_dir ' + model_folder + ' --result_dir . --test_files ' + output_file) 
    p=os.system(str)

    with open('test.predicted.json', mode='r', encoding='utf-8') as f:
        results = json.load(f)

    return json.dumps(results) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5060, debug=True, threaded=True)
