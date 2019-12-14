# coding=utf-8

import math
import os
import random
import jieba
import utils.conf as conf

conv_path = conf.source_data_utils.resource_data #原始语料路径
 
if not os.path.exists(conv_path):
    print('原始语料路径 conv_path 不存在')
    exit()

"""
可读取多轮数据，默认一问一答的方式进行对话
"""
convs = []  # 用于存储对话的列表
with open(conv_path, encoding='utf-8') as f:
    one_conv = [] # 存储一次完整对话
    for line in f:
        line = line.strip('\n').replace('/', '') # 去除换行符，并将原文件中已经分词的标记去掉，重新用结巴分词.
        if line == '':
            continue
        if line[0] == conf.source_data_utils.e:
            if one_conv:
                convs.append(one_conv)
            one_conv = []
        elif line[0] == conf.source_data_utils.m:
            one_conv.append(line[2:]) # 将一次完整的对话存储下来

print(convs[:10])

"""
遍历所有对话，按字切分
"""
ask = []        # 问
response = []   # 答

for conv in convs:
    if len(conv) == 1:
        continue
    if len(conv) % 2 != 0:  # 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
        conv = conv[:-1]
    for i in range(len(conv)):
        if i % 2 == 0:
            conv[i]=" ".join(list(conv[i])) 
            ask.append(conv[i]) # 因为i是从0开始的，因此偶数行为发问的语句，奇数行为回答的语句
        else:
            conv[i]=" ".join(list(conv[i]))
            response.append(conv[i])
"""
将数据分为训练集(train encoder， train decoder)和测试集(test encoder, test decoder)
"""
def convert_seq2seq_files(questions, answers, TESTSET_SIZE):
    train_enc = open(conf.source_data_utils.train_enc,'w')  # 问
    train_dec = open(conf.source_data_utils.train_dec,'w')  # 答
    test_enc  = open(conf.source_data_utils.test_enc, 'w')  # 问
    test_dec  = open(conf.source_data_utils.test_dec, 'w')  # 答
 
    test_index = random.sample([i for i in range(len(questions))], TESTSET_SIZE) # 
 
    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i % 1000000 == 0:
            print(len(range(len(questions))), '处理进度：', i)
 
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
 
convert_seq2seq_files(ask, response,conf.source_data_utils.TEST_SIZE)


