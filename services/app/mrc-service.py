# -*- coding:utf-8 -*-
# use flask to deploy mrc service
# 文档过长时使用 lda，文档过短时使用向量 

from __future__ import unicode_literals

import jieba
import os
import json
import codecs
import flask
import pickle
import argparse

import sys 
sys.path.append("./answer_select/")

from flask import (Flask,render_template,url_for, jsonify, send_from_directory, request, flash)
from flask_cors import cross_origin, CORS
import datetime, time, json, codecs, os, xlrd
from werkzeug.utils import secure_filename
from document_retrieval.retrieval_jd import text_encode, sentences_encode, eculd, annoy
from document_retrieval.retrieval_jd import elastic_search_by_qa
from answer_select.rc_model import RCModel 
from answer_select.dataset import BRCDataset

# parameters for deploy
root_dir = '/export/mrc_flask_deploy'
model_folder = os.path.join(root_dir, "models/model_mrc/bidaf")
vocab_dir = os.path.join(root_dir, "vocab")

build_test_file = "build.test.json"  # 构造的测试数据名称，用于进入rc reader 输出结果 
result_dir = '.' # rc reader 产生的结果输出在当前路径 

app = Flask(__name__)
CORS(app)
app.secret_key='service in inter-credit'

# args for rcmodel
# str=('python ' + root_dir +'solutions/response/mrc/document-retrieval/run.py --predict --algo BIDAF --vocab_dir ' + vocab_dir + ' --model_dir ' + model_folder + ' --result_dir . --test_files ' + build_test_file) 
# os.system(str)
parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
parser.add_argument('--predict', action='store_true', help='predict the answers for test set with trained model')

model_settings = parser.add_argument_group('model settings')
model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF', help='choose the algorithm to use')
model_settings.add_argument('--embed_size', type=int, default=300, help='size of the embeddings')
model_settings.add_argument('--hidden_size', type=int, default=150, help='size of LSTM hidden units')
model_settings.add_argument('--max_p_num', type=int, default=5, help='max passage num in one sample')
model_settings.add_argument('--max_p_len', type=int, default=500, help='max length of passage')
model_settings.add_argument('--max_q_len', type=int, default=60, help='max length of question')
model_settings.add_argument('--max_a_len', type=int, default=200, help='max length of answer')

train_settings = parser.add_argument_group('train settings')
train_settings.add_argument('--optim', default='adam', help='optimizer type')
train_settings.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
train_settings.add_argument('--weight_decay', type=float, default=0, help='weight decay')
train_settings.add_argument('--dropout_keep_prob', type=float, default=1, help='dropout keep rate')
train_settings.add_argument('--batch_size', type=int, default=128, help='train batch size')
train_settings.add_argument('--epochs', type=int, default=10, help='train epochs')

path_settings = parser.add_argument_group('path settings')
path_settings.add_argument('--test_files', nargs='+',default=[build_test_file], help='list of files that contain the preprocessed test data')
path_settings.add_argument('--vocab_dir', default=vocab_dir, help='the dir to save vocabulary')
path_settings.add_argument('--model_dir', default=model_folder, help='the dir to store models')
path_settings.add_argument('--result_dir', default=result_dir, help='the dir to output the results')
path_settings.add_argument('--log_path', help='path of the log file. If not set, logs are printed to console')

args = parser.parse_args()

with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    vocab_file = pickle.load(fin)
rc_model = RCModel(vocab_file, args)
rc_model.restore(model_dir=args.model_dir, model_prefix='BIDAF')


@app.route('/')
def index():
    return("welcome to MRC test " + "*" * 100 )

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def run():
    input_text = request.args.get("inputText")
    es_flag = request.args.get("esFlag", type=bool, default=True)
    semantic_flag = request.args.get("seFlag", type=bool, default=True)
    mrc_flag = request.args.get("mrcFlag", type=bool, default=True)
    # query nlu
    ###
    query = input_text.strip()
    segmented_query = list(jieba.cut(query))
    ###
    # dureader 相关参数
    ###
    question_type = "DESCRIPTION"
    fact_or_opinion = "FACT"
    question_id = 0

    ### 
    # document retireval
    ###

    # es retrieval
    es_answers = elastic_search_by_qa(query)

    # semantic retieval by Embedding
    # all_answers = obj.Get_All_Answer()
    # query_encode = text_encode(query)
    # answers_encode = sentences_encode(all_answers)
    # semantic_answers = annoy(answers_encode, query_encode, all_answers)

    # 汇总es 结果 和 semantic 结果
    all_retrieval_answers = list(set(es_answers))

    print('all retrieval answers', all_retrieval_answers)

    # all_retrieval_answers = list(set(es_answers + semantic_answers))

    ###
    # 构造mrc 模型需要的数据
    ###
    documents = []
    for answer in all_retrieval_answers:
        documents.append({"paragrahs" : [answer], 
                        "segmented_paragraphs":[ list(jieba.cut(answer))],
                        "tile" : "标题",
                        "segmented_tile":["标题"]})
    merge_info = {"question":query,
            "segmented_question":segmented_query,
            "question_id":question_id,
            "question_type":question_type,
            "fact_or_opinion":fact_or_opinion,
            "documents": documents
            }

    with open(build_test_file, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(merge_info, ensure_ascii=False) + "\n")
    ###
    # 运行mrc 模型，输出结果，当前使用脚本直接出结果，
    ###
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, test_files=args.test_files)
    brc_data.convert_to_ids(vocab_file)

    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                            pad_id=vocab_file.get_id(vocab_file.pad_token), 
                                            shuffle=False)
    rc_model.evaluate(test_batches, result_dir=args.result_dir, result_prefix='test.predicted')

    with codecs.open('test.predicted.json', mode='r', encoding='utf-8') as f:
        results = json.load(f)
    return_json = {"answer":results['answers'], "query":query, "retrieval_documents": es_answers}
    # return_json = {"answer":results['answers'], "query":query, "es_retrieval_documents": es_answers, "semantic_retrieval_documents":semantic_answers}

    return json.dumps(return_json, ensure_ascii=False)

