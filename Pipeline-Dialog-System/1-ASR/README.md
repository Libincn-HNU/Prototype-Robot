[TOC]

# paper list

+ Language Model:
  - [CTC](http://www.infocomm-journal.com/dxkx/CN/article/openArticlePDFabs.jsp?id=166970)
  - [baidu](http://proceedings.mlr.press/v48/amodei16.pdf)
+ Language Model:
  - [self-attention](https://arxiv.org/abs/1706.03762)
  - [CBHG](https://github.com/crownpku/Somiao-Pinyin)

# ASR project for chinese
+ [project1](https://github.com/nl8590687/ASRT_SpeechRecognition)
  - issue：开放的模型有问题，不能识别语音
  - speech model: CNN + LSTM/GRU + CTC
  - language model: 最大熵隐马尔可夫模型
+ [baidu PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech)
  - release model [Baidu Internal Mandarin Dataset](https://deepspeech.bj.bcebos.com/demo_models/baidu_cn1.2k_model.tar.gz), [Aishell Dataset](https://deepspeech.bj.bcebos.com/mandarin_models/aishell_model.tar.gz)
  - release language model [0.13 billion n-grams](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm), [3.7 billion n-grams](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm)

# Data Set
+ 清华大学THCHS30中文语音数据集
  - [data_thchs30.tgz](http://cn-mirror.openslr.org/resources/18/data_thchs30.tgz)
+ Free ST Chinese Mandarin Corpus
  - [ST-CMDS-20170001_1-OS.tar.gz](http://cn-mirror.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz)
+ AIShell-1 开源版数据集
  - [data_aishell.tgz](http://cn-mirror.openslr.org/resources/33/data_aishell.tgz)