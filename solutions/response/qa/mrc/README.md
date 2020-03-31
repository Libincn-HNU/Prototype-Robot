# Machine Reading Comprehension Summary

![20200330204239](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330204239.png)

## 0. 背景

+ 机器阅读理解，广泛的范围是指构建机器阅读文档，并理解相关内容，通过不断的阅读和理解来建立逻辑能力，来回答相关问题
+ 现有的一些阅读理解的解决方案主要是针对(文档，问题，答案)这种数据格式进行构建

## 1. 相关任务和数据集

### 1.1 完形填空

+ CNN/Daily Mail
+ [CMRC-2017](https://arxiv.org/pdf/1709.08299.pdf)
  + cloze track

### 1.2 多项选择

+ MC-Test
+ RACE
  + 本文开放出的数据集为中国中学生英语阅读理解题目
  + 给定一篇文章和5道4选1的题目，包括了28000+ passages 和 100,000问题
+ MultiRC
+ Open-BookQA
  + allen ai
  + http://data.allenai.org/OpenBookQA.
  + https://arxiv.org/pdf/1809.02789.pdf

### 1.3 片段抽取

+ SQuAD 1.0
+ SQuAD 2.0
+ NewsQA
+ SearchQA
+ NarrativeQA
+ CMRC-2018 Span-Extraction
  + https://hfl-rc.github.io/cmrc2018/
  + https://github.com/ymcui/cmrc2018

### 1.4 句子填空

+ CMRC 2019
  + https://hfl-rc.github.io/cmrc2019/

### 1.5 事实类问答

+ WebQA ： 针对事实类问答，一共42k question 556k evidence，答案都是单个实体，候选文章较短，都不超过5句话

### 1.6 自由作答

+ 答案可能在原文中出现，也可能在需要根据原文进行推理和归纳总结

+ CoQA
+ MS-MARCO
  + 来自Bing 的搜索日志
  + 英文数据集
  + 10W个问题 20篇不同的文档
+ DuReader
  + 中文
  + 百度搜索和百度知道的数据

## 3. 解决方案

+ pass

## 4. Close Type

### 4.0 Teach Machine to Read and Comprehension

+ https://arxiv.org/abs/1506.03340
+ 一维匹配
+ 二维匹配
+ 推理

## 5. single span extract

### 5.1 BiDAF

+ 原文和源码
  + https://arxiv.org/abs/1611.01603
  + https://github.com/allenai/bi-att-flow
+ 思路
![20200330194903](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330194903.png)
  + 传统的阅读理解方法中将段落表示成固定长度的vector，而BIDAF将段落vector流动起来，利用双向注意力流建模段落和问题之间的关系，得到一个问题感知的段落表征（即段落vector的值有段落本身和当前问题决定，不同的问题会产生不同的vector）
  + 每一个时刻， 仅仅对当前 问题 和段落进行计算，并不依赖上一时刻的attention，使得后边的attetion不受之前错误attention的影响(有待细化）
  + 计算C2Q(context2query) 和 Q2C(query2context), 认为两者相互促进
+ Demo : 基于问题感知的段落表征
  + {context, query} = {汉堡好吃, 吃啥呢} = {H, U}
  + 对应的embedding 为， 向量维度为3(为了便于演示，每个字的向量的元素都相同）
  ![20200330194949](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330194949.png)
  + 计算相似矩阵 H^T * U 
  ![20200330195002](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330195002.png)
  + C2Q ( 对query  进行编码）
  [ 0.36*吃 + 0.42*的 + 0.48*呢,  
    0.54*吃 + 0.63*的 + 0.72*呢,
    0.72*吃 + 0.84*的 + 0.96*呢
    0.9*吃 + 1.05*的 + 1.2*呢]
  + Q2C ( 对context 进行编码）
    + 取每一行的最大值 [0.48, 0.72, 0.96, 1.2] 作为权重，对原始query 做加权和， 并重复T次
    [0.48*汉 + 0.72*堡 + 0.96*好 + 1.2*吃] * T 
  + 拼接 H， C2Q， Q2C ， 并进行一定映射 得到 基于当前query 和 context 的 段落表示 G
  + G在通过LSTM得到M，使用G和M经过MLP 预测起始位置和终止位置 
+ 实验结果
  ![20200330195115](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330195115.png)

### 5.2 Match LSTM

+ pass

### 5.3 AoA

+ pass

## 6. multi-span extract

+ Tag-based Multi-Span Extraction in Reading Comprehension

+ A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning

## 7. multi-paragraph

+ yyHaker：《多篇章阅读理解的Deep Cascade Model》(2018AAAI)-paper阅读笔记

## 8. multi-task

### 8.1 Multi-task Learning with Sample Re-weighting for Machine Reading Comprehension

### 8.2 The Natural Language Decathlon Multitask Learning as Question Answering

+ 将10个NLP任务转为阅读理解任务，具体包括question answering，machine translation，summarization，natural language inference，sentiment analysis，semantic role labeling，relation extraction，goal-oriented dialogue，semantic parsing和commonsense pronoun resolution
+ 作者提出了一个MQAN（Multitask Question Answering Network，类二维匹配模型）来联合训练这些任务。实验结果表明借助多个任务学到的共享信息能有效提高任务效果

### 8.3 Unifying Question Answering and Text Classification via Span Extraction

+ 将Question-Answering和Text Classification问题统一转换为阅读理解问题，借助辅助数据信息提升两者的效果

### 8.4 Entity-Relation Extraction as Multi-turn Question Answering

+ 提出将实体-关系抽取问题转换为多轮阅读理解问题

## 9. Knowledge Enhance

+ xingluxi：基于知识的机器阅读理解论文列表
  + https://zhuanlan.zhihu.com/p/88207389
+ Y.Shu：【WSDM 2019】基于知识图谱嵌入的问答系统
  + https://zhuanlan.zhihu.com/p/86745138
+ IndexFziQ/KMRC-Papers
  + https://github.com/IndexFziQ/KMRC-Papers

+ Probing Prior Knowledge Needed in Challenging Chinese Machine Reading Comprehension
  + 文章构建了一个中文阅读理解数据集C3，并探索了语言信息、领域和通用知识对中文阅读理解的影响。图15实验结果表明语言信息和通用知识对中文阅读理解有一定的提升作用

## 10. dataset-transfor

+ MULTIQA An Empirical Investigation of Generalization and Transfer in Reading Comprehension
  + 探索了不同阅读理解数据间的迁移特性

## 11. open-domain

+ DrQA
+ ORQA
  + https://zhuanlan.zhihu.com/p/35755367

## 最新进展

+ 基于知识的机器阅读理解
+ 不可回答问题
+ 多段式机器阅读理解
+ 对话问答

## 未解决问题

+ 外部知识的整合
+ MRC系统的鲁棒性
+ 给定上下文的局限性
+ 推理能力不足

## Tools

+ Sogou Machine Reading Comprehension， SMRC
  + https://www.leiphone.com/news/201905/Vne0qMUXttiSr8z4.html?viewType=#

## Challenge

+ squad 2.0
+ allenai
  + https://leaderboard.allenai.org/drop/submissions/public
+ baidu
  + https://ai.baidu.com/broad/leaderboard?dataset=dureader

## Survey

+ https://github.com/thunlp/RCPapers
+ Neural Machine Reading Comprehension: Methods and Trends
  + https://arxiv.org/abs/1907.01118v1
  + [MRC综述: Neural MRC: Methods and Trends](https://zhuanlan.zhihu.com/p/75825101)
+ A STUDY OF THE TASKS AND MODELS IN MACHINE READING COMPREHENSION
  + https://arxiv.org/pdf/2001.08635.pdf

## Reference

+ 最AI的小PAI：机器阅读理解探索与实践
+ https://stacks.stanford.edu/file/druid:gd576xb1833/thesis-augmented.pdf
+ Danqi Chen: From Reading Comprehension to Open-Domain Question Answering
+ [后bert时代的机器阅读理解](https://zhuanlan.zhihu.com/p/68893946)
+ [DGCNN 在机器阅读理解上的应用](https://zhuanlan.zhihu.com/p/35755367)