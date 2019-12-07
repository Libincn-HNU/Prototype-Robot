
<!-- TOC -->

- [Target](#target)
- [Dataset](#dataset)
  - [中文数据集](#%e4%b8%ad%e6%96%87%e6%95%b0%e6%8d%ae%e9%9b%86)
  - [中文其他数据集](#%e4%b8%ad%e6%96%87%e5%85%b6%e4%bb%96%e6%95%b0%e6%8d%ae%e9%9b%86)
  - [英文其他数据集](#%e8%8b%b1%e6%96%87%e5%85%b6%e4%bb%96%e6%95%b0%e6%8d%ae%e9%9b%86)
    - [Cornell Movie Dialogs：电影对话数据集，下载地址：http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html](#cornell-movie-dialogs%e7%94%b5%e5%bd%b1%e5%af%b9%e8%af%9d%e6%95%b0%e6%8d%ae%e9%9b%86%e4%b8%8b%e8%bd%bd%e5%9c%b0%e5%9d%80httpwwwcscornelleducristiancornellmovie-dialogscorpushtml)
    - [Ubuntu Dialogue Corpus：Ubuntu日志对话数据，下载地址：https://arxiv.org/abs/1506.08909](#ubuntu-dialogue-corpusubuntu%e6%97%a5%e5%bf%97%e5%af%b9%e8%af%9d%e6%95%b0%e6%8d%ae%e4%b8%8b%e8%bd%bd%e5%9c%b0%e5%9d%80httpsarxivorgabs150608909)
    - [OpenSubtitles：电影字幕，下载地址：http://opus.lingfil.uu.se/OpenSubtitles.php](#opensubtitles%e7%94%b5%e5%bd%b1%e5%ad%97%e5%b9%95%e4%b8%8b%e8%bd%bd%e5%9c%b0%e5%9d%80httpopuslingfiluuseopensubtitlesphp)
    - [Twitter：twitter数据集，下载地址：https://github.com/Marsan-Ma/twitter_scraper](#twittertwitter%e6%95%b0%e6%8d%ae%e9%9b%86%e4%b8%8b%e8%bd%bd%e5%9c%b0%e5%9d%80httpsgithubcommarsan-matwitterscraper)
    - [Papaya Conversational Data Set：基于Cornell、Reddit等数据集重新整理之后，好像挺干净的，下载链接：https://github.com/bshao001/ChatLearner](#papaya-conversational-data-set%e5%9f%ba%e4%ba%8ecornellreddit%e7%ad%89%e6%95%b0%e6%8d%ae%e9%9b%86%e9%87%8d%e6%96%b0%e6%95%b4%e7%90%86%e4%b9%8b%e5%90%8e%e5%a5%bd%e5%83%8f%e6%8c%ba%e5%b9%b2%e5%87%80%e7%9a%84%e4%b8%8b%e8%bd%bd%e9%93%be%e6%8e%a5httpsgithubcombshao001chatlearner)
  - [JDDC](#jddc)
  - [Person-Chat](#person-chat)
  - [DSTC](#dstc)
    - [DSTC1](#dstc1)
    - [DSTC2 and DSTC3](#dstc2-and-dstc3)
    - [DSTC4](#dstc4)
    - [DSTC5](#dstc5)
    - [DSTC6](#dstc6)
    - [DSTC7](#dstc7)
    - [DSTC8](#dstc8)
  - [Ubuntu Dialogue Corpus](#ubuntu-dialogue-corpus)
  - [Goal-Oriented Dialogue Corpus](#goal-oriented-dialogue-corpus)
  - [Standford](#standford)
  - [Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems](#frames-a-corpus-for-adding-memory-to-goal-oriented-dialogue-systems)
  - [Multi WOZ](#multi-woz)
  - [Stanford Multi-turn Multi-domain](#stanford-multi-turn-multi-domain)
- [Metric](#metric)
  - [不是安全回答](#%e4%b8%8d%e6%98%af%e5%ae%89%e5%85%a8%e5%9b%9e%e7%ad%94)
  - [回答具有连续性](#%e5%9b%9e%e7%ad%94%e5%85%b7%e6%9c%89%e8%bf%9e%e7%bb%ad%e6%80%a7)
  - [词重叠评价指标](#%e8%af%8d%e9%87%8d%e5%8f%a0%e8%af%84%e4%bb%b7%e6%8c%87%e6%a0%87)
    - [BLEU](#bleu)
    - [ROUGE](#rouge)
    - [METEOR](#meteor)
  - [词向量评价指标](#%e8%af%8d%e5%90%91%e9%87%8f%e8%af%84%e4%bb%b7%e6%8c%87%e6%a0%87)
    - [Greedy Matching](#greedy-matching)
    - [Embedding Average](#embedding-average)
    - [Vector Extrema](#vector-extrema)
  - [perplexity困惑度](#perplexity%e5%9b%b0%e6%83%91%e5%ba%a6)
- [Solutions](#solutions)
  - [Pipeline](#pipeline)
    - [ASR](#asr)
    - [NLU](#nlu)
    - [DM](#dm)
    - [NLG](#nlg)
    - [TTS](#tts)
  - [NLG](#nlg-1)
    - [Problem](#problem)
      - [个性的一致性](#%e4%b8%aa%e6%80%a7%e7%9a%84%e4%b8%80%e8%87%b4%e6%80%a7)
      - [安全回答](#%e5%ae%89%e5%85%a8%e5%9b%9e%e7%ad%94)
      - [不能指代消解](#%e4%b8%8d%e8%83%bd%e6%8c%87%e4%bb%a3%e6%b6%88%e8%a7%a3)
    - [Seq2seq](#seq2seq)
    - [Transformer2Transformer](#transformer2transformer)
    - [SeqGAN](#seqgan)
    - [CycleGAN](#cyclegan)
  - [IR-Bot](#ir-bot)
    - [DSSM](#dssm)
      - [预处理](#%e9%a2%84%e5%a4%84%e7%90%86)
      - [表示层](#%e8%a1%a8%e7%a4%ba%e5%b1%82)
      - [匹配层](#%e5%8c%b9%e9%85%8d%e5%b1%82)
      - [优缺点](#%e4%bc%98%e7%bc%ba%e7%82%b9)
    - [ARC-I and ARC-II](#arc-i-and-arc-ii)
    - [Match Pyramid](#match-pyramid)
    - [SMN](#smn)
    - [DMN](#dmn)
    - [基于检索的闲聊系统的实现](#%e5%9f%ba%e4%ba%8e%e6%a3%80%e7%b4%a2%e7%9a%84%e9%97%b2%e8%81%8a%e7%b3%bb%e7%bb%9f%e7%9a%84%e5%ae%9e%e7%8e%b0)
  - [FAQ](#faq)
    - [KBQA](#kbqa)
  - [Task-Bot](#task-bot)
- [Reference](#reference)
  - [Links](#links)
  - [Papers](#papers)
    - [Knowledge Aware Conversation Generation with Explainable Reasoing ever Augmented Graphs](#knowledge-aware-conversation-generation-with-explainable-reasoing-ever-augmented-graphs)
    - [Vocabulary Pyramid Network: Multi-Pass Encoding and Decoding with Multi-Level Vocabularies for Response Generation](#vocabulary-pyramid-network-multi-pass-encoding-and-decoding-with-multi-level-vocabularies-for-response-generation)
    - [Personalizing Dialogue Agents: I have a dog, do you have pets too?](#personalizing-dialogue-agents-i-have-a-dog-do-you-have-pets-too)
  - [A Survey of Available Corpora for Building Data-Driven Dialogue Systems](#a-survey-of-available-corpora-for-building-data-driven-dialogue-systems)
    - [A Neural Conversation Model](#a-neural-conversation-model)
    - [Neural Response Generation via GAN with an APProximate Embedding Layer](#neural-response-generation-via-gan-with-an-approximate-embedding-layer)
    - [Deep Reinforcement Learning for Dialogue Generation](#deep-reinforcement-learning-for-dialogue-generation)
  - [Projects](#projects)
    - [JDDC](#jddc-1)
    - [Chatbot](#chatbot)
    - [DST](#dst)
    - [Rasa](#rasa)
    - [Task](#task)
    - [Others](#others)
  - [Tricks](#tricks)
    - [More Deep](#more-deep)
    - [Beam Search](#beam-search)
    - [Pointer Generator](#pointer-generator)
    - [HERD/VHERD/AMI](#herdvherdami)
    - [DRL](#drl)
    - [Deep Reinforcement Learning for Dialogue Generation](#deep-reinforcement-learning-for-dialogue-generation-1)
    - [seqGAN](#seqgan)
    - [构建聊天机器人：检索、seq2seq、RL、SeqGAN](#%e6%9e%84%e5%bb%ba%e8%81%8a%e5%a4%a9%e6%9c%ba%e5%99%a8%e4%ba%ba%e6%a3%80%e7%b4%a2seq2seqrlseqgan)
    - [小姜机器人](#%e5%b0%8f%e5%a7%9c%e6%9c%ba%e5%99%a8%e4%ba%ba)
  - [Books](#books)

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

### Cornell Movie Dialogs：电影对话数据集，下载地址：http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
### Ubuntu Dialogue Corpus：Ubuntu日志对话数据，下载地址：https://arxiv.org/abs/1506.08909
+ UDC 1.0 100W 多轮对话数据
### OpenSubtitles：电影字幕，下载地址：http://opus.lingfil.uu.se/OpenSubtitles.php
### Twitter：twitter数据集，下载地址：https://github.com/Marsan-Ma/twitter_scraper
### Papaya Conversational Data Set：基于Cornell、Reddit等数据集重新整理之后，好像挺干净的，下载链接：https://github.com/bshao001/ChatLearner

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
 

# Metric

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


## NLG

### Problem
#### 个性的一致性
+ Adversarial Learning for Neural Dialogue Generation 
    + 李纪为
#### 安全回答
+ 在seq2seq方法中问题尤为明显，
#### 不能指代消解

### Seq2seq
+ https://blog.csdn.net/Irving_zhang/article/details/79088143
+ https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md
+ https://zhuanlan.zhihu.com/p/29075764

### Transformer2Transformer

### SeqGAN

### CycleGAN



## IR-Bot
+ 主流的方法分为两类，一种是弱相关模型，包括DSSM，ARC-I等方法，另一种是强相关模型，包括ARC-II， MatchPyramid，DeepMatch等算法，两种方法最主要的区别在于对句子<X,Y> 的建模不同，前者是单独建模，后者是联合建模

### DSSM

#### 预处理
+ 英文 word hanshing
  + 以三个字母 切分英文单词，转化后为30k
+ 中文 子向量 15k 个左右常用字

#### 表示层
+ 原始的DSSM　用　BOW ，　后续的其他方法（CNN－DSSM　和　LSTM-DSSM 会有改进）
+ 多层DNN 进行信息表示

#### 匹配层

+ https://www.cnblogs.com/wmx24/p/10157154.html

![DSSM-1.PNG](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/DSSM-1.PNG)

#### 优缺点
+ 优点：DSSM 用字向量作为输入既可以减少切词的依赖，又可以提高模型的泛化能力，因为每个汉字所能表达的语义是可以复用的。另一方面，传统的输入层是用 Embedding 的方式（如 Word2Vec 的词向量）或者主题模型的方式（如 LDA 的主题向量）来直接做词的映射，再把各个词的向量累加或者拼接起来，由于 Word2Vec 和 LDA 都是无监督的训练，这样会给整个模型引入误差，DSSM 采用统一的有监督训练，不需要在中间过程做无监督模型的映射，因此精准度会比较高。
+ 缺点：上文提到 DSSM 采用词袋模型（BOW），因此丧失了语序信息和上下文信息。另一方面，DSSM 采用弱监督、端到端的模型，预测结果不可控。

### ARC-I and ARC-II
+ https://arxiv.org/pdf/1503.03244.pdf

### Match Pyramid

### SMN

### DMN

### 基于检索的闲聊系统的实现
+ 使用检索引擎（如ES）对所有预料进行粗粒度的排序
  + 使用Okapi BM２５　算法

+ 使用匹配算法对答案进行精排

## FAQ

### KBQA

## Task-Bot


# Reference

## Links

- [AIComp Top](https://github.com/linxid/AICompTop)
- [Robot Design](https://github.com/qhduan/ConversationalRobotDesign)
- [sizhi bot](https://github.com/ownthink/robot]
- [home assistant](https://github.com/home-assistant/home-assistant)
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


### 构建聊天机器人：检索、seq2seq、RL、SeqGAN
+ https://blog.csdn.net/Young_Gy/article/details/76474939

### 小姜机器人
+ https://blog.csdn.net/rensihui/article/details/89418850
+ 模版/检索/生成

## Books
+ 自然语言处理实践-聊天机器人原理与应用
