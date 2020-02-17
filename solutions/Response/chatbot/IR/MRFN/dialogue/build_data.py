import pickle
with open('str-history-true-false.pkl', mode='rb') as f:
    results = pickle.load(f)
    history, true_utt, false_utt = results['history'], results['true_utt'], results['false_utt']

print('read data done')

span = 100000
history = history[ 10*span: 10*span + 1000 ]
true_utt = true_utt[ 10*span: 10*span + 1000 ]
false_utt = false_utt[ 10*span: 10*span + 1000 ]

print('build data done')

results = []

"""
tmp_his [[0, 0, '约', '你', '妹', '呀', '。']]
str_his ["[0, 0, '约', '你', '妹', '呀', '。']"]

"""


def convert(inputs):
    results = []
    for item in inputs:
        results.append(str(item))

    return results

import jieba

for tmp_his, tmp_true, tmp_false in zip(history, true_utt, false_utt):
    
    # 拼成整句
    str_his = ["".join(convert(tmp)) for  tmp in tmp_his]
    str_true = ''.join(convert(tmp_true))
    str_false = ''.join(convert(tmp_false))


    # 分词
    str_his = [" ".join(list(jieba.cut(tmp))) for  tmp in str_his]
    str_true = ' '.join(list(jieba.cut(str_true)))
    str_false = ' '.join(list(jieba.cut(str_false)))

    text_true = str('1' + '\t' +  '\t'.join(str_his) + '\t' + str_true )
    text_false = str('0' + '\t' +  '\t'.join(str_his) + '\t' + str_false)

    # print('text_true', text_true)
    # print('text_false', text_false)

    results.append(text_true + '\n')
    results.append(text_false + '\n')

#for tmp_his, tmp_true, tmp_false in zip(history, true_utt, false_utt):
#    
#    str_his = ["".join(convert(tmp)) for  tmp in tmp_his]
#    str_true = ''.join(convert(tmp_true))
#    str_false = ''.join(convert(tmp_false))
#    text_true = str('1' + '\t' +  '\t'.join(str_his) + '\t' + str_true )
#    text_false = str('0' + '\t' +  '\t'.join(str_his) + '\t' + str_false)
#
#    #print('text_true', text_true)
#    #print('text_false', text_false)
#
#    results.append(text_true + '\n')
#    results.append(text_false + '\n')

with open('train.txt', mode='w', encoding='utf-8') as f:
    f.writelines(results)
