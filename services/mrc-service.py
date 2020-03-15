# -*- coding:utf-8 -*-
from __future__ import unicode_literals

import jieba
import os
import json
import codecs
import flask

from flask import (Flask,render_template,url_for, jsonify, send_from_directory, request, flash)
from flask_cors import cross_origin, CORS
import datetime, time, json, codecs, os, xlrd
from werkzeug.utils import secure_filename
from es_jd.es_tool_jd import ElasticObj

root_dir = '/export/mrc_flask_deploy/'


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
    obj = ElasticObj("brc_index_name", "brc_index_type")
    query = input_text.strip()
    segmented_question = list(jieba.cut(query))
    question_type = "DESCRIPTION"
    fact_or_opinion = "FACT"
    question_id = 0
    answers = obj.Get_Data_By_Body(query)
    documents = []
    return_documents = []

    for answer in answers:
        documents.append({"paragrahs" : [answer], 
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
    # print("&"*100)
    # print(merge_info)

    build_test_file = "build.test.json" 
    model_folder = root_dir + "services/models/model_mrc/bidaf"
    vocab_dir = root_dir + "solutions/response/mrc/document-retrieval/data/"
    with open(build_test_file, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(merge_info, ensure_ascii=False) + "\n")

    str=('python ' + root_dir +'solutions/response/mrc/document-retrieval/run.py --predict --algo BIDAF --vocab_dir ' + vocab_dir + ' --model_dir ' + model_folder + ' --result_dir . --test_files ' + build_test_file) 
    p=os.system(str)

    with codecs.open('test.predicted.json', mode='r', encoding='utf-8') as f:
        results = json.load(f)

    print('results', results)

    return_json = {"answer":results['answers'], "question":query, "documents": answers}

    return json.dumps(return_json, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5060, debug=True, threaded=True)
