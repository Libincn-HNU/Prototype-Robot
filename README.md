# Summary of Dialogue System

![20200315174742-2020-3-15-17-47-42](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315174742-2020-3-15-17-47-42)

# Papers
+ [martian-ai/Awesome-Paper](https://github.com/martian-ai/Awesome-Paper/blob/master/chatbot-papers-candidates.md)

<<<<<<< HEAD
## Framework
=======

# Corpus & Preprocessing
+ See in 'corpus' folder
+ include 
    + chatbot
    + mrc
    + skill

# Related work
## 小冰
+ https://arxiv.org/pdf/1812.08989v1.pdf

![20200315174630-2020-3-15-17-46-31](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315174630-2020-3-15-17-46-31)
​
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
​
## Alex
## Mitsuku
## Cleverbot
## DeepBot

# Framework
>>>>>>> 168739def04157cd77828ec6c00d8e2502ff367d
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

"""
1.baseline:
GBDT (domain classification) Hypotheses Ranking for Robust Domain Classification And Tracking in Dialogue Systems
2. RNNs
CONTEXTUAL SPOKEN LANGUAGE UNDERSTANDING USING RECURRENT NEURAL NETWORKS
CONTEXTUAL DOMAIN CLASSIFICATION IN SPOKEN LANGUAGE UNDERSTANDING SYSTEMS USING RECURRENT NEURAL NETWORK
Context Sensitive Spoken Language Understanding using Role Dependent LSTM layers
3. MemNets:
End-to-End Memory Networks with Knowledge Carryover for Multi-Turn Spoken Language Understanding
Sequential Dialogue Context Modeling for Spoken Language Understanding
4. CNNs on DA(dialog act classification task)
Using Context Information for Dialog Act Classification in DNN Framework
"""

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

<<<<<<< HEAD
+ 阅读理解

+ 对话管理

=======
# Reference
xingluxi：基于知识的机器阅读理解论文列表
Ted Li：rasa的component，policy，action的自定义开发
BreezeDeus：微软小冰对话机器人架构
李永彬：小蜜团队万字长文：讲透对话管理模型研究最新进展
https://arxiv.org/abs/1711.01731
夕小瑶：认真的聊一聊对话系统（任务型、检索式、生成式对话论文与工具串讲）
>>>>>>> 168739def04157cd77828ec6c00d8e2502ff367d
