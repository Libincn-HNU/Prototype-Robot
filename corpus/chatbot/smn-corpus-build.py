
with open("/Users/sunhongchao/Documents/craft/Awesome-Corpus/done/multi-answer-pure.txt", mode='w', encoding='utf-8') as f:
    f.writelines(tmp_answer_list)

with open("/Users/sunhongchao/Documents/craft/Awesome-Corpus/done/multi-question-pure.txt", mode='w', encoding='utf-8') as f:
    f.writelines(tmp_query_list)


import random
import numpy as np

# 获取字典  word2idx
vocab = open('/export/home/sunhongchao1/Prototype-Robot/solutions/FAQ/NLG/seqGAN/gen_data/vocab5000.all', 'r', encoding='utf-8', newline='\n', errors='ignore')
word2idx = {}

for idx, value in enumerate(vocab):
    word2idx[value.strip()] = idx

print(word2idx)

# 获取 embedding matrix
embedding_path = '/export/home/sunhongchao1/Workspace-of-NLU/resources/Tencent_AILab_ChineseEmbedding.txt'

fin = open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
word2vec = {}
for line in fin:
    tokens = line.rstrip().split(' ') 
    if tokens[0] in word2idx.keys():
        word2vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')

embedding_matrix = np.zeros((len(word2idx), 200))
unknown_words_vector = np.random.rand(200)

for word, idx in word2idx.items():
    if word in word2vec.keys():
        embedding_matrix[idx] = word2vec[word]
    else:
        embedding_matrix[idx] = unknown_words_vector


history, true_utt, false_utt = [], [], []

"""
读取单轮数据
"""
with open("/export/home/sunhongchao1/Prototype-Robot/corpus/dialogue/new_corpus.txt", mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for idx in range(0, len(lines)-3, 3):
        history.append([ [ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in lines[idx][2:]]]) # 加入的是一个list
        true_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in lines[idx+1][2:]])
        tmp_utt =lines[random.choice(range(len(lines)//3)) + 1 ][2:]
        false_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in tmp_utt ])

"""
读取多轮数据
默认一问一答
"""

def random_choice_false_response(lines):

    tmp_flag = True
    while tmp_flag:
        idx = random.choice(len(lines))
        
        if lines[idx].startwith('M'):
            tmp_flag = False

    return lines[idx]

with open("/export/home/sunhongchao1/Prototype-Robot/corpus/dialogue/new_corpus_multi.txt", mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    tmp_text_list = []
    for idx in range(lines):
        if lines[idx].startwith('E'):
            tmp_history = []
            for item in tmp_text_list[:-1]:
                tmp_history.append([word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in item])    
            history.append(tmp_history)

            true_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in tmp_test_list[-1]])

            tmp_utt = random_choice_false_response(lines)
            false_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in tmp_utt ])
            tmp_text_list = []
        elif lines[idx].startwith('M'):
            tmp_text_list.append(lines[idx][2:])
        else:
            print('not start with E or M error')

import pickle
results = {'history':history, 'true_utt':true_utt, 'false_utt':false_utt}
save_file = open("results.pkl","wb")
pickle.dump(results, save_file)
save_file.close()

embedding = {'embedding_matrix':embedding_matrix}
save_file = open("embedding_matrix.pkl","wb")
pickle.dump(embedding, save_file)
save_file.close()


## 句法分析 == >>> 不满足特定条件的句法树 删除

# from snownlp import SnowNLP

# tmp_list = []
# for item in tmp_list:
#     s = SnowNLP(item)
#     s.summary(limit=4) # limit 为最长多少句
