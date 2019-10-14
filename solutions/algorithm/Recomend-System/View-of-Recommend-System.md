# View of Recommender System

### A review on deep learning for recommender systems: challenges and remedies
+ 一篇利用深度学习做推荐系统的综述
+ 从深度学习模型方面对问题进行了分类
+ 从推荐系统研究方面对问题进行了分类

### Explainable Recommendation: A survey and New Perspectives
+ 可解释性问答系统 相关最新研究的总结


### Real-time Personalization using Embeddings for Search Ranking Airbnb
+ Airbnb KDD2018 Applied Data Science Track Best Paper
+ 使用word embedding 的思路训练 待选民宿房间 的 embedding 和 用户的 embedding
+ 在此基础上进行推荐和搜索

### Learning from History and Present: Next-item Recommendation via Discriminaatively Exploiting User Behaviors
+ 中科大 京东
+ SIGKDD 2018
+ 现有的序列化推荐方法往往对消费者的短期行为进行分析,没有充分考虑用户的长期偏好及偏好的动态变化
+ 该文基于用户行为区别, 提取了一个基于商品推荐的任务的全新BINN(Behavior Intensive Neural Network)模型, 该模型包括一个Item Embedding 和 两个RNN
+ Item Embedding 对用户产生的Item 序列运用类似skip-gram的模型, 两个RNN分别用于捕获用户的当前偏好和历史偏好

###  Evaluation of Session-based Recommendation Algorithms
  - Recommender System
  - 本文系统地介绍了 Session-based Recommendation，主要针对 baseline methods, nearest-neighbor techniques, recurrent neural networks 和 (hybrid) factorization-based methods 等 4 大类算法进行介绍。
  - 此外，本文使用 RSC15、TMALL、ZALANDO、RETAILROCKET、8TRACKS 、AOTM、30MUSIC、NOWPLAYING、CLEF 等 7 个数据集进行分析，在 Mean Reciprocal Rank (MRR)、Coverage、Popularity bias、Cold start、Scalability、Precision、Recall 等指标上进行比较。
  - 代码链接
  	- https://www.dropbox.com/sh/7qdquluflk032ot/AACoz2Go49q1mTpXYGe0gaANa?dl=0

### Recommendation in Heterogeneous Information Networks Based on Generalized Random Walk Model and  Bayesian Personalized Ranking
+ 北京大学
+ WSDM 2018
+ 基于异构信息网络(HIN), 具有协同过滤, 内容过滤, 上下文感知推荐
+ 现有各类方法中的关键在于如何正确的设置在异构信息网络中各种link的权重
+ 该文提出了一种基于贝叶斯的个性化排序(BRP)的机器学习方法-HeteLearn, 来学习异构信息网络中的link权重,并用来测试个性化推荐任务
