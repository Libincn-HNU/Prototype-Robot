# Multi-Representation Fusion Network (MRFN)

This is an implementation of [Multi-Representation Fusion Network for Multi-turn Response Selection in Retrieval-based Chatbots, WSDM 2019].


## Requirements
* Ubuntu 16.04
* Tensorflow 1.4.0
* Python 2.7
* NumPy

## Usage
To download and preprocess the data, run

```bash
# download ubuntu corpus and word/char dictionaries and pre-trained embeddings 
sh download.sh
# preprocess the data
python data_utils_record.py
```

All hyper parameters are stored in config.py. To train, run

```bash
python main.py --log_root=logs_ubuntu --batch_size=100
```

To evaluate the model, run
```bash
python evaluate.py --log_root=logs_ubuntu --batch_size=100
```


# 需要构造的数据
+ 语料
label \t u1 \t u2 ... un \t r
+ word2idx
+ word embedding metric
+ char2idx
+ char embedding metric

# 测试
+ 使用es 获得候选集
+ 使用 构造成tfrecord 的格式 