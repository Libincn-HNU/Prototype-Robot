
<!-- TOC -->

1. [Workspace of Conversation AI](#workspace-of-conversation-ai)
2. [Target](#target)
3. [Dataset](#dataset)
    1. [DSTC](#dstc)
        1. [DSTC1](#dstc1)
        2. [DSTC2 and DSTC3](#dstc2-and-dstc3)
        3. [DSTC4](#dstc4)
        4. [DSTC5](#dstc5)
        5. [DSTC6](#dstc6)
        6. [DSTC7](#dstc7)
        7. [DSTC8](#dstc8)
    2. [Ubuntu Dialogue Corpus](#ubuntu-dialogue-corpus)
    3. [Goal-Oriented Dialogue Corpus](#goal-oriented-dialogue-corpus)
    4. [Standford](#standford)
    5. [Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems](#frames-a-corpus-for-adding-memory-to-goal-oriented-dialogue-systems)
    6. [Multi WOZ](#multi-woz)
    7. [Stanford Multi-turn Multi-domain](#stanford-multi-turn-multi-domain)
    8. [A Survey of Available Corpora for Building Data-Driven Dialogue Systems](#a-survey-of-available-corpora-for-building-data-driven-dialogue-systems)
4. [Resource](#resource)
5. [Metric](#metric)
6. [Solutions](#solutions)
    1. [Chat-Bot](#chat-bot)
    2. [IR-Bot](#ir-bot)
    3. [QA-Bot](#qa-bot)
    4. [KB-Bot](#kb-bot)
    5. [Task-Bot](#task-bot)
    6. [Pipeline](#pipeline)
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
    3. [Projects](#projects)
        1. [Personalizing Dialogue Agents: I have a dog, do you have pets too?](#personalizing-dialogue-agents-i-have-a-dog-do-you-have-pets-too)
        2. [Seq2seq code build](#seq2seq-code-build)
        3. [DeepQA](#deepqa)
        4. [RasaHQ](#rasahq)
        5. [NLG/GAN](#nlggan)
        6. [Dual Training](#dual-training)
        7. [More Deep](#more-deep)
        8. [A Neural Conversation Model](#a-neural-conversation-model)

<!-- /TOC -->

# Workspace of Conversation AI

# Target
+ Step 1. Collect current papers, corpus and projects
+ Step 2. Pipeline model
+ Step 3. End2End model

# Dataset

+ [chatbot corpus chinese](https://github.com/codemayq/chaotbot_corpus_Chinese)

+ [Dialog corpus](https://github.com/candlewill/Dialog_Corpus)

+ [chat_corpus](https://github.com/Marsan-Ma-zz/chat_corpus)

+ [bAbI](https://github.com/facebook/bAbI-tasks)

+ [dstc8-reddit-corpus](https://github.com/microsoft/dstc8-reddit-corpus)

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
  
## A Survey of Available Corpora for Building Data-Driven Dialogue Systems
  
- 把所有的数据集按照不同类别进行分类总结，里面涵盖了很多数据集
  - [链接](http://link.zhihu.com/?target=https%3A//docs.google.com/spreadsheets/d/1SJ4XV6NIEl_ReF1odYBRXs0q6mTkedoygY3kLMPjcP8/pubhtml)

# Resource

+ pass



# Metric



# Solutions

## Chat-Bot

## IR-Bot

## QA-Bot

## KB-Bot

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

## Projects

+ [2018 JDDC对话大赛亚军解决方案 Dialog-System-with-Task-Retrieval-and-Seq2seq](https://github.com/Dikea/Dialog-System-with-Task-Retrieval-and-Seq2seq)
+ [seq2seq chatbot](https://github.com/lc222/seq2seq_chatbot)
+ [Task-Oriented-Dialogue-Dataset-Survey](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey)
+ [jddc_solution_4th](https://github.com/zengbin93/jddc_solution_4th)
+ [jddc_baseline_tfidf](https://github.com/SimonJYang/JDDC-Baseline-TFIDF)
+ [jddc_baseline_seq2seq](https://github.com/SimonJYang/JDDC-Baseline-TFIDF)
+ [transformer_chatbot](https://github.com/atselousov/transformer_chatbot)
+ [Seq2Seq_Chatbot_QA](https://github.com/qhduan/Seq2Seq_Chatbot_QA)
+ [rasa_chatbot](https://github.com/zqhZY/_rasa_chatbot)
+ [TC-bot](https://github.com/MiuLab/TC-Bot)
+ [DNN-DST](https://github.com/CallumMain/DNN-DST)
+ [chatbot-MemN2N-tf](https://github.com/vyraun/chatbot-MemN2N-tensorflow)
+ [DST](https://github.com/voicy-ai/DialogStateTracking)

+ [LatticeLSTM](https://github.com/jiesutd/LatticeLSTM)
+ [Text CLF](https://github.com/jatana-research/Text-Classification)
+ [Text CLF cnn-rnn-tf](https://github.com/gaussic/text-classification-cnn-rnn)
+ [text_classification](https://github.com/brightmart/text_classification)

### Personalizing Dialogue Agents: I have a dog, do you have pets too?

- 本文是 Facebook AI Research 发表于 NIPS 2018 的工作。论文根据一个名为 PERSONA-CHAT 的对话数据集来训练基于 Profile 的聊天机器人，该数据集包含超过 16 万条对话。
- 本文致力于解决以下问题：
- 聊天机器人缺乏一致性格特征
  - 聊天机器人缺乏长期记忆
  - 聊天机器人经常给出模糊的回应，例如 I don't know
- 数据集链接
- https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat

### Seq2seq code build
+ https://blog.csdn.net/Irving_zhang/article/details/79088143

https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md

### DeepQA
+ Seq2Seq
+ https://zhuanlan.zhihu.com/p/29075764

### RasaHQ

### NLG/GAN

### Dual Training

### More Deep
- 在可以收敛的情况下，尽可能使用更深的模型
- 参考CV 领域的一些做法
- https://zhuanlan.zhihu.com/p/35317776
- https://zhuanlan.zhihu.com/p/29967933
- https://blog.csdn.net/Irving_zhang/article/details/79088143

### A Neural Conversation Model
+ https://arxiv.org/abs/1506.05869

