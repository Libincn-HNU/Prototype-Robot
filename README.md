---
noteId: "28910f1019bd11eabcd5bf51d3f5c99a"
tags: []

---

<!-- TOC -->

- [Target](#target)
- [Doing](#doing)
- [ToDo List](#todo-list)
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
    - [Text Geneartion from Knowledge Graphs with Graph Transformers](#text-geneartion-from-knowledge-graphs-with-graph-transformers)
  - [Projects](#projects)
    - [JDDC](#jddc)
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

# Doing
+ seqGan 和其他 生成类方法
+ IR bot KB bot 等其他方法

# ToDo List



# Reference

## Links

- [AIComp Top](https://github.com/linxid/AICompTop)
- [Robot Design](https://github.com/qhduan/ConversationalRobotDesign)
- [sizhi bot](https://github.com/ownthink/robot]
- [home assistant](https://github.com/home-assistant/home-assistant)
- 评价指标
    - https://blog.csdn.net/liuchonge/article/details/79104045
- GAN + 文本生成的一些论文

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

### Text Geneartion from Knowledge Graphs with Graph Transformers
+ https://blog.csdn.net/TgqDT3gGaMdkHasLZv/article/details/100190240

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
