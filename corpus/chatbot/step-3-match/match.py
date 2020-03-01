import pickle

input_filename = 'corpus-step-1.pkl'
mz_input_filename = 'match-zoo-corpus-top-1w.csv'

with open(input_filename, 'rb') as f:
    all_conv_list = pickle.load(f)

print('all conv list', len(all_conv_list))

cut_number = 1000

all_conv_list = all_conv_list[:cut_number] 
print('cut ', cut_number)

all_conv_list = [tmp for tmp in all_conv_list if len(tmp) > 1]

print(all_conv_list[:10])

print('new conv list build done')

"""

匹配过滤

"""

# 构造数据
import csv,random
import numpy as np
out = open(mz_input_filename, mode='w', encoding='utf-8', newline='')
csv_write = csv.writer(out,dialect='excel')

pre_text = ["text_left", "text_right", "label"]
csv_write.writerow(pre_text)
for idx, item in enumerate(all_conv_list):

    if len(item) < 2:
        print("len item < 2")
        continue
    
    if len(item[0]) < 4:
        print('len item[0] < 4')
        continue

    csv_write.writerow([str(item[0]), str(item[1]), '1'])
    count = 0
    while count < 5:
        tmp = random.choice(all_conv_list)
        if len(tmp) < 2:
            print('error tmp', tmp)
            continue
        elif len(tmp[1]) < 6:
            print("tmp [1] < 6")
            continue
        else:
            csv_write.writerow([str(item[0]), str(tmp[1]), '0']) # 改进方式，不适用随机文本，改为使用 AA 相似文本
            count = count + 1

print('*'*100)
print('build date done')
print('*'*100)

# 读数据
import matchzoo as mz
import pandas as pd
print(mz.__version__)
data_pack = mz.pack(pd.read_csv(mz_input_filename)) # 必须有 text_left, text_right, label, 其他的会自动补齐
data_pack.relation['label'] = data_pack.relation['label'].astype('float32')
frame = data_pack.frame

### 定义任务，包含两种，一个是Ranking，一个是classification
task = mz.tasks.Ranking()
### 准备数据，数据在源码中有，不确定在pip安装的是否存在
### train_raw是matchzoo中自定的数据格式	matchzoo.data_pack.data_pack.DataPack
train_raw = data_pack # mz.datasets.toy.load_data(stage='train', task=task)
test_raw = data_pack # mz.datasets.toy.load_data(stage='test', task=task)

### 数据预处理，BasicPreprocessor为指定预处理的方式，在预处理中包含了两步：fit,transform
### fit将收集一些有用的信息到preprocessor.context中，不会对输入DataPack进行处理
### transformer 不会改变context、DataPack,他将重新生成转变后的DataPack.
### 在transformer过程中，包含了Tok

# Tokenize => Lowercase => PuncRemoval等过程，这个过程在方法中应该是可以自定义的
preprocessor = mz.preprocessors.BasicPreprocessor()
# preprocessor = mz.preprocessors.DSSMPreprocessor()
preprocessor.fit(train_raw, verbose=0)  ## init preprocessor inner state.
train_processed = preprocessor.transform(train_raw, verbose=0)
test_processed = preprocessor.transform(test_raw, verbose=0)

### 创建模型以及修改参数（可以使用mz.models.list_available()查看可用的模型列表）
model = mz.models.DenseBaseline()
model.params['task'] = task
model.params['mlp_num_units'] = 128
model.params.update(preprocessor.context)
model.params.completed()
model.build()
model.compile()
model.backend.summary()

### 训练, 评估, 预测
x, y = train_processed.unpack()
test_x, test_y = test_processed.unpack()
model.fit(x , y, batch_size=512, epochs=5)
model.evaluate(test_x,test_y)
results = model.predict(test_x)


#print(len(results))
#tmp_list = [str(item) for item in frame]
#print(tmp_list[:10])
#print(tmp_list[-10:])


### 保存模型
model.save('step-2-mz-model')
#loaded_model = mz.load_model('my-model')

