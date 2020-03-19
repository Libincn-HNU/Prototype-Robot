import random
import pickle
import jieba
import numpy as np

# 读取原始文本，构建一个较小的embedding, pickle 大于2G 的无法load
with open('corpus-step-1.pkl', mode='rb') as f:
    corpus = pickle.load(f)

"""
corpus
"""

all_occurence_word_set = set()

for tmp_list in corpus:
    for line in tmp_list:
        cuts = list(jieba.cut(line))
        for item in cuts:
            if len(item) > 1:
                all_occurence_word_set.add(item)

print('all occurence word set build done')

word2idx = {}
# 获取字典  word2idx
with open('/export/home/sunhongchao1/Prototype-Robot/corpus/word2idx_tencent.pkl','rb') as f:
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

embedding_matrix = np.zeros((len(all_occurence_word_set) + 1, 200))
#embedding_matrix = np.zeros((len(word2idx) + 22751 + 1, 200))
print(len(word2idx))
print(len(all_occurence_word_set))
unknown_words_vector = np.random.rand(200)

idx = 1
for word in all_occurence_word_set:
    if word in word2vec.keys():
        embedding_matrix[idx] = word2vec[word]
    else:
        embedding_matrix[idx] = unknown_words_vector

    idx = idx + 1

embedding = {'word_embedding_matrix':embedding_matrix}
save_file = open("word_embedding_matrix.pkl","wb")
pickle.dump(embedding, save_file, protocol=4)
save_file.close()
