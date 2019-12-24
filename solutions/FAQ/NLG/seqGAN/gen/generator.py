from __future__ import division
from __future__ import print_function
import math
import os
import random
import sys
import time
import pickle
import heapq
import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import utils.data_utils as data_utils
import utils.conf as conf
import gen.gen_model as seq2seq_model
from tensorflow.python.platform import gfile

sys.path.append('../utils')


def read_data_from_file(config, source_path, target_path, max_size=None):
    """
    读取数据，用于产生训练集和测试集， 产生带有 bucket 的数据
    """
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading disc_data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets): #[bucket_id, (source_size, target_size)]
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def create_model(session, gen_config, forward_only, name_scope, initializer=None):
    """
    加载 ckpt 并 产生模型

    注意: 在不同的机器上，需要修改 ckpt 所在的路径 和 其中的 checkpoint 文件中的模型路径
    """
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = seq2seq_model.Seq2SeqModel(gen_config,  name_scope=name_scope, forward_only=forward_only)
        # gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
        # gen_ckpt_dir = os.path.abspath(os.path.join("/Users/sunhongchao/Documents/craft/Prototype-Robot/solutions/NLG/seqGAN/gen_data/", "checkpoints"))
        ckpt = tf.train.get_checkpoint_state("/Users/sunhongchao/Documents/craft/Prototype-Robot/solutions/NLG/seqGAN/gen_data/checkpoints/")
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print(" * " * 50)
            print("Reading Gen model parameters from %s" % ckpt.model_checkpoint_path)
            print(" * " * 50)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print(" * " * 50)
            print("Created Gen model with fresh parameters.")
            print(" * " * 50)
            gen_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            session.run(tf.variables_initializer(gen_global_variables))
        return model

def prepare_data(gen_config):
    """
    功能:
        创建字典 和 训练集 测试集
    输入: 
        gen_config
    输出:
        vacab : [(word1,idx1), (word2, idx2), ...]
        revacab : [(idx1, word1), (idx2, word2), ...]
        dev_set : [[], [], [], []] bucket1 到 bucket4 的数据
        train_set : [[], [], [], []] bucket1 到 bucket4 的数据 
    """
    train_path = os.path.join(gen_config.train_dir, "train")
    voc_file_path = [train_path+".answer", train_path+".query"]
    vocab_path = os.path.join(gen_config.train_dir, "vocab%d.all" % gen_config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size) # 读取 输入文本，按字切分，统计字出现的次数，并将前 vocab_size 的 写入到文件
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path) # 读取写入到 文件的 top vocab_size 的数据，创建 两种格式 的 字典

    # 读取 训练数据 和 验证数据，并转化为 token index
    print("Data Tokenizer : Preparing Chitchat gen_data in %s" % gen_config.train_dir)
    train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(gen_config.train_dir, vocab, gen_config.vocab_size)

    print ("Reading development and training gen_data (limit: %d)." % gen_config.max_train_data_size)
    dev_set = read_data_from_file(gen_config, dev_query, dev_answer)
    train_set = read_data_from_file(gen_config, train_query, train_answer, gen_config.max_train_data_size) #数据格式：train_set[[ [[source],[target]],[[source],[target]] ],....]  最外层的维度为bucket的个数

    return vocab, rev_vocab, dev_set, train_set

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob

def pretrain(gen_config):
    """
    生成器 预训练
    """

    "pretrain for generator"
    print("*" * 50, " Generator Pretrain: begin", "*"*50)
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)

    for b_set in train_set:
        print("length is ", len(b_set))

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.emb_dim))
        model = create_model(sess,gen_config,forward_only=False,name_scope="genModel")

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        #previous_losses = []

        print("*" * 50, " Generator Pretrain: train 10000 step ", "*"*50)
        while current_step<10000:
            # Choose a bucket according to disc_data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            # encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(train_set, bucket_id, gen_config.batch_size)
            encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch(train_set, bucket_id, gen_config.batch_size)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gen_config.steps_per_checkpoint * 50 == 0:

                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step-time %.2f perplexity " "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                print("current_step: %d, save model" %(current_step))
                gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                checkpoint_path = os.path.join(gen_ckpt_dir, "chitchat.model")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

    print("*" * 50, " Generator : end", "*"*50)