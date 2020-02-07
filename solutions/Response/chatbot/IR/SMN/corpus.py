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
读取单轮数据 和 多轮的混合数据
"""
def tmp_utt_choice(lines:list):
    return lines[0]

def deal_conv(inputs:list):
    history.append([ [ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in inputs[-1:] ]]) # 加入的是一个list
    true_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in inputs[-1] ])

    tmp_utt = tmp_utt_choice(inputs)
    false_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in tmp_utt ])
    

with open("/export/home/sunhongchao1/Prototype-Robot/corpus/dialogue/in_use/merge_weibo_xhj_singlemulti.txt", mode='r', encoding='utf-8') as f:

    tmp_list = []
    for line in f:
        if len(line.strip()) is 0:
            if len(tmp_list) > 1:
                deal_conv(tmp_list)
                tmp_list = []
            else:
                print('error in tmp list')
        else:
            tmp_list.append(line)

import pickle
results = {'history':history, 'true_utt':true_utt, 'false_utt':false_utt}
save_file = open("results.pkl","wb")
pickle.dump(results, save_file)
save_file.close()

embedding = {'embedding_matrix':embedding_matrix}
save_file = open("embedding_matrix.pkl","wb")
pickle.dump(embedding, save_file)
save_file.close()
