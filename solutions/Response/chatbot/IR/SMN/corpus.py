import random
import pickle
import numpy as np

word2idx = {}
# 获取字典  word2idx
with open('/export/home/sunhongchao1/Prototype-Robot/corpus/char2idx_tencent.pkl','rb') as f:
    word2idx = pickle.load(f)

print('load word2idx done')

# 获取 embedding matrix
embedding_path = '/export/home/sunhongchao1/Workspace-of-NLU/resources/Tencent_AILab_ChineseEmbedding.txt'

fin = open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
word2vec = {}
for line in fin:
    tokens = line.rstrip().split(' ') 
    if tokens[0] in word2idx.keys():
        word2vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')

embedding_matrix = np.zeros((len(word2idx) + 1, 200))
print(len(word2idx))
unknown_words_vector = np.random.rand(200)

for word, idx in word2idx.items():
    if word in word2vec.keys():
        embedding_matrix[idx] = word2vec[word]
    else:
        embedding_matrix[idx] = unknown_words_vector


history, true_utt, false_utt, all_response = [], [], [], []

"""
读取单轮数据 和 多轮的混合数据
"""

def deal_conv(inputs:list):
    # history.append([ [ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in inputs[-1:] ]]) # 加入的是一个list
    history.append([ [ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in inputs[-1:] ]]) # 加入的是一个list
    true_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in inputs[-1] ])
    all_response.append(inputs[-1])
    false_utt_all = random.choice(all_response) 
    false_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in false_utt_all])
    
import pickle
with open("/export/home/sunhongchao1/Prototype-Robot/corpus/corpus-step-1.pkl", mode='rb') as f:
    conv_list = pickle.load(f) # 读到的是一个个conv list 

    tmp_list = []
    for item in conv_list:
        print('item conv list', item)
        deal_conv(item)

import pickle
results = {'history':history, 'true_utt':true_utt, 'false_utt':false_utt}
save_file = open("history-ture-false.pkl","wb")
pickle.dump(results, save_file)
save_file.close()

embedding = {'embedding_matrix':embedding_matrix}
save_file = open("embedding_matrix.pkl","wb")
pickle.dump(embedding, save_file)
save_file.close()
