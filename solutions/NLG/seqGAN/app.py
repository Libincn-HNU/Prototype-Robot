from flask import Flask, render_template, request, make_response
from flask import jsonify
import flask

import sys
import time  
import hashlib
import threading
import jieba
import utils.conf as conf
import gen.generator as generator
import utils.conf as conf

"""
定义心跳检测函数
"""

def heartbeat():
    print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()

app = Flask(__name__,static_url_path="/static") 

@app.route('/message', methods=['POST', 'GET'])
def reply():
    inputs = flask.request.args.get('input')
    res_msg = execute.decoder_online(sess, conf.gen_config, model, vocab, rev_vocab, inputs )
    # res_msg=generator.get_predicted_sentence(sess, inputs ,vocab,model,conf.gen_config.beam_size,conf.gen_config.buckets)
    
    res_msg = res_msg.replace('_UNK', '^_^') #将unk值的词用微笑符号袋贴
    res_msg=res_msg.strip()
    
    if res_msg == ' ':  
      res_msg = '请与我聊聊天吧'

    return jsonify( { 'text': res_msg } )

@app.route("/")
def index(): 
    return render_template("index.html")

import tensorflow as tf
import execute

sess = tf.Session()
sess, model, vocab, rev_vocab = execute.init_session(sess,conf.gen_config)

if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8808, debug=True) 
