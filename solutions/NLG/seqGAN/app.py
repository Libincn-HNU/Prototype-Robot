from flask import Flask, render_template, request, make_response
from flask import jsonify
import flask

import os
import sys
import time  
import hashlib
import threading
import jieba
import utils.conf as conf
import gen.generator as generator
import utils.conf as conf
import utils.data_utils as data_utils
import gen.generator as gens


def init_session(sess, gen_config):
    """
    decode online 中使用
    """
    model = gens.create_model(sess, gen_config, forward_only=True, name_scope="genModel")
    vocab_path = os.path.join('/Users/sunhongchao/Documents/craft/Prototype-Robot/solutions/NLG/seqGAN/gen_data', "vocab%d.all" % gen_config.vocab_size)
    # vocab_path = os.path.join('/Users/sunhongchao/Documents/craft/Prototype-Robot/solutions/NLG/seqGAN', gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
    return sess ,model, vocab, rev_vocab


def decoder_online(sess, gen_config, model, vocab,rev_vocab, inputs):
    
    token_ids = data_utils.sentence_to_token_ids(inputs, vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
    #bucket_id = min([i for i in xrange(len(train_buckets_scale))
       #             if train_buckets_scale[i] > random_number_01])
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights , _, _ = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id,gen_config.batch_size)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
    
    # If there is an EOS symbol in outputs, cut them at that point.
    tokens = []
    resps = []
    for seq in output_logits:
        token = []
        for t in seq:
            token.append(int(np.argmax(t, axis=0)))
        tokens.append(token)
        tokens_t = []
        for col in range(len(tokens[0])):
            tokens_t.append([tokens[row][col] for row in range(len(tokens))])

        for seq in tokens_t:
            if data_utils.EOS_ID in seq:
                resps.append(seq[:seq.index(data_utils.EOS_ID)][:gen_config.buckets[bucket_id][1]])
            else:
                resps.append(seq[:gen_config.buckets[bucket_id][1]])
    for resp in resps:
        resq_str= " ".join([tf.compat.as_str(rev_vocab[output]) for output in resp])
    return resq_str

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
    res_msg = decoder_online(sess, conf.gen_config, model, vocab, rev_vocab, inputs )
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
sess, model, vocab, rev_vocab = init_session(sess,conf.gen_config)

if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8808, debug=True) 
