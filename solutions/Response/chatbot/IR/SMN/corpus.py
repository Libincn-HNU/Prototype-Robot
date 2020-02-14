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

def deal_conv_str(inputs:list):
    tmp_history = inputs[:-1]
    tmp_true= inputs[-1]

    his_result = []
    for item in tmp_history:
        his_result.append([ tmp if tmp in word2idx.keys() else 0 for tmp in item])

    history.append(his_result) # 加入的是一个list

    true_utt.append([ tmp if tmp in word2idx.keys() else 0 for tmp in tmp_true])

    all_response.append(tmp_true)
    tmp_false = random.choice(all_response) # 此处有待优化 
    false_utt.append([ tmp if tmp in word2idx.keys() else 0 for tmp in tmp_false])


"""
读取单轮数据 和 多轮的混合数据
"""

def deal_conv(inputs:list):
    tmp_history = inputs[:-1]
    tmp_true= inputs[-1]

    his_result = []
    for item in tmp_history:
        his_result.append([ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in item])

    history.append(his_result) # 加入的是一个list

    true_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in tmp_true])

    all_response.append(tmp_true)
    tmp_false = random.choice(all_response) # 此处有待优化 
    false_utt.append([ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in tmp_false])

    
import pickle
with open("/export/home/sunhongchao1/Prototype-Robot/corpus/corpus-step-1.pkl", mode='rb') as f:
    conv_list = pickle.load(f) # 读到的是一个个conv list 
    count = 0 
    tmp_list = []
    for item in conv_list:
        if len(item) > 1:
            # deal_conv(item)
            deal_conv_str(item)
        else:
            count = count + 1
            print('wrong format count {}/{}'.format(count, len(conv_list)),
                  end='\r')

print('history top 110', history[100:110])
print('true top 110', true_utt[100:110])
print('false top 110', false_utt[100:110])


import pickle
results = {'history':history[:10000], 'true_utt':true_utt[:10000],
           'false_utt':false_utt[:10000]}
#results = {'history':history, 'true_utt':true_utt, 'false_utt':false_utt}
save_file = open("str-history-true-false-top10000.pkl","wb")
# save_file = open("str-history-true-false.pkl","wb")
pickle.dump(results, save_file)
save_file.close()

embedding = {'embedding_matrix':embedding_matrix}
save_file = open("embedding_matrix.pkl","wb")
pickle.dump(embedding, save_file)
save_file.close()
