
<!-- TOC -->

1. [Target](#target)
2. [Dataset](#dataset)
    1. [中文数据集](#中文数据集)
    2. [中文其他数据集](#中文其他数据集)
    3. [英文其他数据集](#英文其他数据集)
    4. [JDDC](#jddc)
    5. [Person-Chat](#person-chat)
    6. [DSTC](#dstc)
        1. [DSTC1](#dstc1)
        2. [DSTC2 and DSTC3](#dstc2-and-dstc3)
        3. [DSTC4](#dstc4)
        4. [DSTC5](#dstc5)
        5. [DSTC6](#dstc6)
        6. [DSTC7](#dstc7)
        7. [DSTC8](#dstc8)
    7. [Ubuntu Dialogue Corpus](#ubuntu-dialogue-corpus)
    8. [Goal-Oriented Dialogue Corpus](#goal-oriented-dialogue-corpus)
    9. [Standford](#standford)
    10. [Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems](#frames-a-corpus-for-adding-memory-to-goal-oriented-dialogue-systems)
    11. [Multi WOZ](#multi-woz)
    12. [Stanford Multi-turn Multi-domain](#stanford-multi-turn-multi-domain)
3. [Resource](#resource)
4. [Metric](#metric)
5. [对话系统中的自然语言生成技术](#对话系统中的自然语言生成技术)
    1. [不是安全回答](#不是安全回答)
    2. [回答具有连续性](#回答具有连续性)
    3. [词重叠评价指标](#词重叠评价指标)
        1. [BLEU](#bleu)
        2. [ROUGE](#rouge)
        3. [METEOR](#meteor)
    4. [词向量评价指标](#词向量评价指标)
        1. [Greedy Matching](#greedy-matching)
        2. [Embedding Average](#embedding-average)
        3. [Vector Extrema](#vector-extrema)
    5. [perplexity困惑度](#perplexity困惑度)
6. [Solutions](#solutions)
    1. [Chat-Bot](#chat-bot)
        1. [Problem](#problem)
            1. [个性的一致性](#个性的一致性)
            2. [安全回答](#安全回答)
            3. [不能指代消解](#不能指代消解)
        2. [Rasa_Bot](#rasa_bot)
        3. [Seq2seq](#seq2seq)
        4. [bi-Transformer](#bi-transformer)
    2. [IR-Bot](#ir-bot)
        1. [SMN](#smn)
        2. [DMN](#dmn)
    3. [QA-Bot](#qa-bot)
        1. [KBQA](#kbqa)
    4. [Task-Bot](#task-bot)
    5. [Pipeline](#pipeline)
        1. [ASR](#asr)
        2. [NLU](#nlu)
        3. [DM](#dm)
        4. [NLG](#nlg)
        5. [TTS](#tts)
7. [Reference](#reference)
    1. [Links](#links)
    2. [Papers](#papers)
        1. [Knowledge Aware Conversation Generation with Explainable Reasoing ever Augmented Graphs](#knowledge-aware-conversation-generation-with-explainable-reasoing-ever-augmented-graphs)
        2. [Vocabulary Pyramid Network: Multi-Pass Encoding and Decoding with Multi-Level Vocabularies for Response Generation](#vocabulary-pyramid-network-multi-pass-encoding-and-decoding-with-multi-level-vocabularies-for-response-generation)
        3. [Personalizing Dialogue Agents: I have a dog, do you have pets too?](#personalizing-dialogue-agents-i-have-a-dog-do-you-have-pets-too)
    3. [A Survey of Available Corpora for Building Data-Driven Dialogue Systems](#a-survey-of-available-corpora-for-building-data-driven-dialogue-systems)
        1. [A Neural Conversation Model](#a-neural-conversation-model)
        2. [Neural Response Generation via GAN with an APProximate Embedding Layer](#neural-response-generation-via-gan-with-an-approximate-embedding-layer)
        3. [Deep Reinforcement Learning for Dialogue Generation](#deep-reinforcement-learning-for-dialogue-generation)
    4. [Projects](#projects)
        1. [JDDC](#jddc-1)
        2. [Chatbot](#chatbot)
        3. [DST](#dst)
        4. [Rasa](#rasa)
        5. [Task](#task)
        6. [Others](#others)
    5. [Tricks](#tricks)
        1. [More Deep](#more-deep)
        2. [Beam Search](#beam-search)
        3. [Pointer Generator](#pointer-generator)
        4. [HERD/VHERD/AMI](#herdvherdami)
        5. [DRL](#drl)
        6. [Deep Reinforcement Learning for Dialogue Generation](#deep-reinforcement-learning-for-dialogue-generation-1)
        7. [seqGAN](#seqgan)
        8. [CycleGAN](#cyclegan)
        9. [构建聊天机器人：检索、seq2seq、RL、SeqGAN](#构建聊天机器人检索seq2seqrlseqgan)
        10. [小姜机器人](#小姜机器人)

<!-- /TOC -->

# Target
+ Step 1. Collect current papers, corpus and projects
+ Step 2. Pipeline model
+ Step 3. End2End model

# Dataset

## 中文数据集

语料名称 | 语料数量 | 语料来源说明 | 语料特点 | 语料样例 | 是否已分词
---|---|---|---|---|---
[chatterbot](https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese) | 560 | 开源项目 | 按类型分类，质量较高  | Q:你会开心的 A:幸福不是真正的可预测的情绪。 | 否
[douban 豆瓣多轮](https://github.com/MarkWuNLP/MultiTurnResponseSelection ) | 352W | 来自北航和微软的paper, 开源项目 | 噪音相对较少，原本是多轮（平均7.6轮）  | Q:烟台 十一 哪 好玩 A:哪 都 好玩 · · · · | 是
[ptt PTT八卦语料](https://github.com/zake7749/Gossiping-Chinese-Corpus) | 40W | 开源项目，台湾PTT论坛八卦版 | 繁体，语料较生活化，有噪音  | Q:为什么乡民总是欺负国高中生呢QQ	A:如果以为选好科系就会变成比尔盖兹那不如退学吧  | 否
qingyun（青云语料） | 10W | 某聊天机器人交流群 | 相对不错，生活化  | Q:看来你很爱钱 	 A:噢是吗？那么你也差不多了 | 否
[subtitle 电视剧对白语料](https://github.com/fateleak/dgk_lost_conv) | 274W | 开源项目，来自爬取的电影和美剧的字幕 | 有一些噪音，对白不一定是严谨的对话，原本是多轮（平均5.3轮）  | Q:京戏里头的人都是不自由的	A:他们让人拿笼子给套起来了了 | 否
tieba 贴吧论坛回帖语料 | 232W | 偶然找到的 | 多轮，有噪音 https://pan.baidu.com/s/1mUknfwy1nhSM7XzH8xi7gQ 密码:i4si  | Q:前排，鲁迷们都起床了吧	A:标题说助攻，但是看了那球，真是活生生的讽刺了 | 否
weibo（微博语料） | 443W | 华为 Noah 实验室 Neural Responding Machine for Short-Text Conversation | 仍有一些噪音  | Q:北京的小纯洁们，周日见。#硬汉摆拍清纯照# A:嗷嗷大湿的左手在干嘛，看着小纯洁撸么。 | 否
[xiaohuangji（小黄鸡语料）](https://github.com/candlewill/Dialog_Corpus) | 45W | 原人人网项目语料 | 有一些不雅对话，少量噪音 | Q:你谈过恋爱么	A:谈过，哎，别提了，伤心..。 | 否


## 中文其他数据集
- 三千万字幕语料
https://link.zhihu.com/?target=http%3A//www.shareditor.com/blogshow/%3FblogId%3D112

- 白鹭时代中文问答语料
    - 白鹭时代论坛问答数据，一个问题对应一个最好的答案。下载链接：https://github.com/Samurais/egret-wenda-corpus
- 微博数据集
    - 华为李航实验室发布，也是论文“Neural Responding Machine for Short-Text Conversation”使用的数据集下载链接：http://61.93.89.94/Noah_NRM_Data/
- 新浪微博数据集
    - 评论回复短句，下载地址：http://lwc.daanvanesch.nl/openaccess.php
    
## 英文其他数据集

Cornell Movie Dialogs：电影对话数据集，下载地址：http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
Ubuntu Dialogue Corpus：Ubuntu日志对话数据，下载地址：https://arxiv.org/abs/1506.08909
OpenSubtitles：电影字幕，下载地址：http://opus.lingfil.uu.se/OpenSubtitles.php
Twitter：twitter数据集，下载地址：https://github.com/Marsan-Ma/twitter_scraper
Papaya Conversational Data Set：基于Cornell、Reddit等数据集重新整理之后，好像挺干净的，下载链接：https://github.com/bshao001/ChatLearner

## JDDC

+ 需要注册才能得到数据集
+ 有待上传

## Person-Chat
    + Facebook
    + 16w 条

## DSTC

  - The Dialog State Tracking Challenge (DSTC) is an on-going series of research community challenge tasks. Each task released dialog data labeled with dialog state information, such as the user’s desired restaurant search query given all of the dialog history up to the current turn. The challenge is to create a “tracker” that can predict the dialog state for new dialogs. In each challenge, trackers are evaluated using held-out dialog data.

### DSTC1

  - DSTC1 used human-computer dialogs in the bus timetable domain. Results were presented in a special session at [SIGDIAL 2013](http://www.sigdial.org/workshops/sigdial2013/). DSTC1 was organized by Jason D. Williams, Alan Black, Deepak Ramachandran, Antoine Raux.
  - Data : https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/#!dstc1-downloads
  - Project:
    - pass
    - pass

### DSTC2 and DSTC3

- DSTC2/3 used human-computer dialogs in the restaurant information domain. Results were presented in special sessions at [SIGDIAL 2014](http://www.sigdial.org/workshops/conference15/) and [IEEE SLT 2014](http://www.slt2014.org/). DSTC2 and 3 were organized by Matthew Henderson, Blaise Thomson, and Jason D. Williams.
- Data : http://camdial.org/~mh521/dstc/
- Project:
  - pass
  - pass

### DSTC4

- DSTC4 used human-human dialogs in the tourist information domain. Results were presented at [IWSDS 2015](http://www.iwsds.org/). DSTC4 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, and Jason D. Williams.
- Data:
  - http://www.colips.org/workshop/dstc4/data.html
- Project:
  - pass

### DSTC5

- DSTC5 used human-human dialogs in the tourist information domain, where training dialogs were provided in one language, and test dialogs were in a different language. Results were presented in a special session at [IEEE SLT 2016](http://www.slt2016.org/). DSTC5 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, Jason D. Williams, and Koichiro Yoshino.
- Data:
  - http://workshop.colips.org/dstc5/data.html
- Project:
  - Pass

### DSTC6

- DSTC6 consisted of 3 parallel tracks:
  - End-to-End Goal Oriented Dialog Learning
  - End-to-End Conversation Modeling
  - Dialogue Breakdown Detection.
- Results will be presented at a workshop immediately after NIPS 2017.
  - DSTC6 is organized by Chiori Hori, Julien Perez, Koichiro Yoshino, and Seokhwan Kim.
- Tracks were organized by Y-Lan Boureau, Antoine Bordes, Julien Perez, Ryuichi Higashinaka, Chiori Hori, and Takaaki Hori.

### DSTC7

### DSTC8

## Ubuntu Dialogue Corpus

- The Ubuntu Dialogue Corpus : A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems, 2015 [[paper\]](http://arxiv.org/abs/1506.08909) [[data\]](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
  
## Goal-Oriented Dialogue Corpus
  
- **(Frames)** Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems, 2016 [[paper\]](https://arxiv.org/abs/1704.00057) [[data\]](http://datasets.maluuba.com/Frames)
- **(DSTC 2 & 3)** Dialog State Tracking Challenge 2 & 3, 2013 [[paper\]](http://camdial.org/~mh521/dstc/downloads/handbook.pdf) [[data\]](http://camdial.org/~mh521/dstc/)
  
## Standford
  
- A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset
- Mihail Eric and Lakshmi Krishnan and Francois Charette and Christopher D. Manning. 2017. Key-Value Retrieval Networks for Task-Oriented Dialogue. In Proceedings of the Special Interest Group on Discourse and Dialogue (SIGDIAL). https://arxiv.org/abs/1705.05414. [pdf]
- https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
- http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip
  - calendar scheduling
- weather information retrieval
  - point-of-interest navigation
  
## Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems

- Maluuba 放出的对话数据集。
- 论文链接：http://www.paperweekly.site/papers/407
  - 数据集链接：http://datasets.maluuba.com/Frames

## Multi WOZ

- https://www.repository.cam.ac.uk/handle/1810/280608
  
## Stanford Multi-turn Multi-domain
  
- 包含三个domain（日程，天气，景点信息），可参考下该数据机标注格式：
  - https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
- 论文citation
  - Key-Value Retrieval Networks for Task-Oriented Dialogue https://arxiv.org/abs/1705.05414
  
- 把所有的数据集按照不同类别进行分类总结，里面涵盖了很多数据集
 
# Resource

+ pass

# Metric


# 对话系统中的自然语言生成技术
- https://zhuanlan.zhihu.com/p/49197552


## 不是安全回答

## 回答具有连续性

## 词重叠评价指标

### BLEU

### ROUGE

### METEOR

## 词向量评价指标

### Greedy Matching

### Embedding Average

### Vector Extrema

## perplexity困惑度

# Solutions

## Chat-Bot

### Problem
#### 个性的一致性
+ Adversarial Learning for Neural Dialogue Generation 
    + 李纪为
#### 安全回答
#### 不能指代消解

### Rasa_Bot
+ 

### Seq2seq
+ https://blog.csdn.net/Irving_zhang/article/details/79088143
+ https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md
+ https://zhuanlan.zhihu.com/p/29075764

### bi-Transformer
 

## IR-Bot

### SMN

### DMN

## QA-Bot

### KBQA

## Task-Bot

## Pipeline

### ASR

- APIs or Tools for free

### NLU

- Domain CLF
  - context based domain clf
- Intent Detection
- Slot Filling
- Joint Learning and Ranking

### DM

- DST
- DPL

### NLG

### TTS


# Reference

## Links

- [AIComp Top](https://github.com/linxid/AICompTop)
- [Robot Design](https://github.com/qhduan/ConversationalRobotDesign)
- [sizhi bot](https://github.com/ownthink/robot]
- [home assistant](https://github.com/home-assistant/home-assistant)
- [textClassifier](https://github.com/jiangxinyang227/textClassifier)
- 评价指标
    - https://blog.csdn.net/liuchonge/article/details/79104045

## Papers

### Knowledge Aware Conversation Generation with Explainable Reasoing ever Augmented Graphs

+ EMNLP 2019 Baidu
+ Tips 
  + 大部分模型 容易出现安全回复和不连贯回复，这是因为仅仅从语料中学习语义而不是借助背景知识
  + 引入结构化信息
    + 利用三元组或者图路径来缩小知识的候选范围并增强模型的泛化能力
    + 但是选出的知识往往是实体或普通词，因而无法为回复生成更加丰富的信息
  + 引入非结构化信息
    + 文本知识(电影评论或者电影剧情)可以为回复的生成提供丰富的参考信息，但是非结构化的表示方案要求模型具有很强的能力来从知识文本集合中进行知识选择或者使用注意力机制
+ 此文综合使用结构化信息和非结构化信息， 提出了基于扩充知识图(Augmented Knowledge Graph)开放域对话生成模型，模型由知识选择和回复生成这两个模块组成

### Vocabulary Pyramid Network: Multi-Pass Encoding and Decoding with Multi-Level Vocabularies for Response Generation

+ seq2seq框架作为文本生成的主流框架，在对话领域已被广泛应用。然而在对话生成时，seq2seq倾向于生成通用的答案，缺乏进一步的润色修饰；***解码的过程中词表大小的限制与逐词解码的方式存在偏置问题***，***且目标端的全局信息无法利用***。如何在解码过程中不断丰富词表，利用全局信息丰富句子内容是本论文的主要研究贡献。
+ 论文链接：https://www.aclweb.org/anthology/P19-1367/ 

### Personalizing Dialogue Agents: I have a dog, do you have pets too?

- 本文是 Facebook AI Research 发表于 NIPS 2018 的工作。论文根据一个名为 PERSONA-CHAT 的对话数据集来训练基于 Profile 的聊天机器人，该数据集包含超过 16 万条对话。
- 本文致力于解决以下问题：
- 聊天机器人缺乏一致性格特征
  - 聊天机器人缺乏长期记忆
  - 聊天机器人经常给出模糊的回应，例如 I don't know
- 数据集链接
- https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat

## A Survey of Available Corpora for Building Data-Driven Dialogue Systems

- (https://arxiv.org/pdf/1512.05742.pdf

### A Neural Conversation Model

+ https://arxiv.org/abs/1506.05869

### Neural Response Generation via GAN with an APProximate Embedding Layer
+ 单轮回答，抑制安全回答

+ 实现  
+ https://github.com/lan2720/GAN-AEL
+ https://github.com/deepanshugarg257/Response-Generation-with-AEL

### Deep Reinforcement Learning for Dialogue Generation
+ https://www.cnblogs.com/jiangxinyang/p/10469860.html

+ 传统的seq2seq 问题
+ 1> 安全回答
+ 2> 使用MLE 容易死循环

## Projects

### JDDC
+ [2018 JDDC对话大赛亚军解决方案 Dialog-System-with-Task-Retrieval-and-Seq2seq](https://github.com/Dikea/Dialog-System-with-Task-Retrieval-and-Seq2seq)
+ [seq2seq chatbot](https://github.com/lc222/seq2seq_chatbot)
+ [jddc_solution_4th](https://github.com/zengbin93/jddc_solution_4th)
+ [jddc_baseline_tfidf](https://github.com/SimonJYang/JDDC-Baseline-TFIDF)
+ [jddc_baseline_seq2seq](https://github.com/SimonJYang/JDDC-Baseline-TFIDF)

### Chatbot
+ [Seq2Seq_Chatbot_QA](https://github.com/qhduan/Seq2Seq_Chatbot_QA)
+ [Awesome-chatbot](https://github.com/fendouai/Awesome-Chatbot)
+ [transformer-chatbot](https://github.com/atselousov/transformer_chatbot)
    + pytorch
+ [chatbot-MemN2N-tf](https://github.com/vyraun/chatbot-MemN2N-tensorflow)
+ [seq2seqchatbots](https://github.com/ricsinaruto/Seq2seqChatbots)
    + 有常见的数据集处理的代码
    + transformer
    

### DST
+ [DNN-DST](https://github.com/CallumMain/DNN-DST)
+ [DST](https://github.com/voicy-ai/DialogStateTracking)

### Rasa
+ [rasa_chatbot_cn](https://github.com/GaoQ1/rasa_chatbot_cn)
+ [_rasa_chatbot](https://github.com/zqhZY/_rasa_chatbot)
+ [rasa_chatbot](https://github.com/zqhZY/_rasa_chatbot)

### Task
+ [Task-Oriented-Dialogue-Dataset-Survey](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey)

### Others
+ [TC-bot](https://github.com/MiuLab/TC-Bot)

## Tricks

### More Deep
- 在可以收敛的情况下，尽可能使用更深的模型
- 参考CV 领域的一些做法
- https://zhuanlan.zhihu.com/p/35317776
- https://zhuanlan.zhihu.com/p/29967933

### Beam Search

### Pointer Generator

### HERD/VHERD/AMI
+ 多轮
+ https://blog.csdn.net/liuchonge/article/details/79237611

Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models(HRED)
A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues(VHRED)
Attention with Intention for a Neural Network Conversation Model(AWI)

### DRL
+ https://blog.csdn.net/liuchonge/article/details/78749623
+ https://zhuanlan.zhihu.com/p/21587758

### Deep Reinforcement Learning for Dialogue Generation

+ https://zhuanlan.zhihu.com/p/21587758

### seqGAN
+ https://github.com/zhaoyingjun/chatbot/blob/master/seq2seqChatbot/seq2seq_model.py
+ rl 对抗训练
+ https://www.jianshu.com/p/b8c3d2a42ba7
+ https://blog.csdn.net/yuuyuhaksho/article/details/87560253

### CycleGAN

### 构建聊天机器人：检索、seq2seq、RL、SeqGAN
+ https://blog.csdn.net/Young_Gy/article/details/76474939

### 小姜机器人
+ https://blog.csdn.net/rensihui/article/details/89418850
+ 模版/检索/生成
