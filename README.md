# Summary of Dialogue System

![ds_pic_1.png](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/ds_pic_1.png)


# Corpus & Preprocessing
+ See in 'corpus' folder
+ include 
    + chatbot
    + mrc
    + skill

# Related work
## 小冰
+ https://arxiv.org/pdf/1812.08989v1.pdf

![ds_xiaoice_1.png](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/ds_xiaoice_1.png)

## DialoGPT
+ https://arxiv.org/abs/1911.00536
+ https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/
+ https://github.com/microsoft/DialoGPT

## PAI
+ 最AI的小PAI：多轮人机对话与对话管理技术探索与实践
+ 最AI的小PAI：CCKS 2019 | 开放域中文KBQA系统
+ 最AI的小PAI：NLP上层应用的关键一环——中文纠错技术简述
+ 最AI的小PAI：人机交互场景下的知识挖掘：对话日志挖掘的核心流程和相关模型
+ 最AI的小PAI：智能问答系统：问句预处理、检索和深度语义匹配技术

## Alex
## Mitsuku
## Cleverbot
## DeepBot

# Framework
+ https://app.gitmind.cn/doc/4f4a6005da14c77e840f13c15dd3af03

# Solutions

## query
+ nlu
    + 分词/词性/实体
    + 句法
    + 关键词
    + 长难句压缩
    + 问题改写
    + 单轮状态识别

## dm
+ up
+ kb
+ state tracking
    + 多轮情感识别
    + 多轮领域识别
    + 多轮意图识别


+ context query understanding
    + 实体链接
    + 指代消解
    + 句子补全
+ user simulation
    + Topic Model
+ agent setting

## response
+ skill
    + task
    Task-Oriented-Dialogue-Dataset-Survey](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey)
+ qa
    + kbqa
    + qbda(mrc)
+ chatbot
    + ir
    + nlg

# Metric
+ 困惑度
+ SSA
+ 生成回复长度
+ 熵
+ BLEU
+ 图灵测试
