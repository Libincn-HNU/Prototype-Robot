<!-- TOC -->

1. [Some Algorithms for NLG](#some-algorithms-for-nlg)
2. [Target](#target)
    1. [单轮对话能生成流畅的回复](#单轮对话能生成流畅的回复)
    2. [多轮对话能结合上下文产生回复](#多轮对话能结合上下文产生回复)
3. [Problem](#problem)
    1. [安全回答](#安全回答)
    2. [个性的一致性](#个性的一致性)
    3. [不能指代消解](#不能指代消解)
4. [Metric](#metric)
    1. [Human judgenment](#human-judgenment)
    2. [Dataset](#dataset)
5. [Seq2seq](#seq2seq)
6. [Transformer2Transformer](#transformer2transformer)
7. [SeqGAN](#seqgan)
8. [CycleGAN](#cyclegan)
9. [GPT2 for Chinese chitchat](#gpt2-for-chinese-chitchat)
10. [Experiments](#experiments)
11. [Reference](#reference)
    1. [Views](#views)
        1. [对话系统中的自然语言生成技术](#对话系统中的自然语言生成技术)
        2. [NLG != 机器写作](#nlg--机器写作)
        3. [多轮检索式对话系统小节](#多轮检索式对话系统小节)
        4. [A Hybrid Retrieval-Generation Neural Conversation Model](#a-hybrid-retrieval-generation-neural-conversation-model)
        5. [QuaSE：Sequence Editing under Quantifiable Guidance](#quasesequence-editing-under-quantifiable-guidance)
        6. [Towards Less Ceneric Responses in Neural Conversation Models: A Statictical Re-weighting Method](#towards-less-ceneric-responses-in-neural-conversation-models-a-statictical-re-weighting-method)

<!-- /TOC -->

# Some Algorithms for NLG

# Target
## 单轮对话能生成流畅的回复
## 多轮对话能结合上下文产生回复

# Problem

## 安全回答
+ 在seq2seq方法中问题尤为明显

## 个性的一致性
+ Adversarial Learning for Neural Dialogue Generation 
    + 李纪为
    
## 不能指代消解

# Metric
## Human judgenment
+ Adequacy
+ Fluency
+ Readability
+ Variation

## Dataset
+ Tourist Information Dataset
+ Restaurant in San Francisco Dataset


# Seq2seq
+ https://blog.csdn.net/Irving_zhang/article/details/79088143
+ https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md
+ https://zhuanlan.zhihu.com/p/29075764

# Transformer2Transformer

# SeqGAN
![seqGAN-1.png](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/seqGAN-1.png)

# CycleGAN

# GPT2 for Chinese chitchat
+ https://github.com/yangjianxin1/GPT2-chitchat
+ 本项目使用GPT2模型对中文闲聊语料进行训练，使用 HuggingFace的transformers实现GPT2模型的编写与训练。
+ 在闲暇时间用 GPT2-Chinese模型训练了几个长文本的生成模型，并且精读了一遍作者的源码，获益匪浅，加深了自己对GPT2生成模型的一些理解，于是将GPT2模型用于闲聊对话的生成，非常感谢作者的分享。
+ 本项目中沿用了原项目中的部分结构和一些命名方式，同时也对很多代码细节做出了自己实现。
+ 解码器的逻辑使用了Temperature、Top-k Sampling和Nucleus Sampling等，可参考论文The Curious Case of Neural Text Degeneration


# Experiments
	前10W 数据	全数据
seq2seq	语句不通顺	安全回答居多（图片已经保存）
seqGAN-19600		答案错误，需要调试
GAN-AEL		
GAN-RL		

# Reference

## Views
### 对话系统中的自然语言生成技术
+ https://zhuanlan.zhihu.com/p/49197552
+ 介绍了 基于模版，基于树， Plan-based， Class-based， Phrase-based， Corpus-based， RNN-base LM， Semantic Conditioned LSTM, Structural NLG(句法树 + 神经网络), Contextual NLG(使用seq2seq, 考虑上下文，适合多轮对话)， Controlled Text Generation(基于GAN 的 NLG), Transfor Learning for NLG(用迁移学习做NLG，可以解决目标领域数据不足的问题，也可以跨语言、个性化等)
+ 以上方法对比
![.png-2019-12-10-16-47-31](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/.png-2019-12-10-16-47-31)

### NLG != 机器写作
+ https://zhuanlan.zhihu.com/p/44149779

### 多轮检索式对话系统小节
+ https://zhuanlan.zhihu.com/p/84163773

### A Hybrid Retrieval-Generation Neural Conversation Model 
+ https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.09068%3Fcontext%3Dcs
+ 分析检索式和生成式的特性，优缺点



![nlp-category-1-2019-12-25-18-30-48](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/nlp-category-1-2019-12-25-18-30-48)

NLG 的一些应用场景和方法
https://arxiv.org/pdf/1906.00500.pdf

Knowledge-based Controllable Writing Generation

https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/83388740

### QuaSE：Sequence Editing under Quantifiable Guidance
+ 作者提出一种新的量化指标引导下的序列编辑模型，可以编辑生成与给定的量化指标相匹配的句子

### Towards Less Ceneric Responses in Neural Conversation Models: A Statictical Re-weighting Method
+ 作者提出一种适用于开放领域对话系统的新型神经对话模型（Neural Conversation Model），旨在解决生成模型中容易产生通用回复（如“我不知道”，“我也是”）的问题
+ 实验分析还不错