# Some Algorithms for NLG

# Target
## 单轮对话能生成流畅的回复
## 多轮对话能结合上下文产生回复

# Problem
## 个性的一致性
+ Adversarial Learning for Neural Dialogue Generation 
    + 李纪为
## 安全回答
+ 在seq2seq方法中问题尤为明显，
## 不能指代消解

# Seq2seq
+ https://blog.csdn.net/Irving_zhang/article/details/79088143
+ https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md
+ https://zhuanlan.zhihu.com/p/29075764

# Transformer2Transformer

# SeqGAN
![seqGAN-1.png](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/seqGAN-1.png)

# CycleGAN

# Metric
## Human judgenment
+ Adequacy
+ Fluency
+ Readability
+ Variation

## Dataset
+ Tourist Information Dataset
+ Restaurant in San Francisco Dataset

# Reference

## Views
### 对话系统中的自然语言生成技术
+ https://zhuanlan.zhihu.com/p/49197552
+ 介绍了 基于模版，基于树， Plan-based， Class-based， Phrase-based， Corpus-based， RNN-base LM， Semantic Conditioned LSTM, Structural NLG(句法树 + 神经网络), Contextual NLG(使用seq2seq, 考虑上下文，适合多轮对话)， Controlled Text Generation(基于GAN 的 NLG), Transfor Learning for NLG(用迁移学习做NLG，可以解决目标领域数据不足的问题，也可以跨语言、个性化等)
+ 以上方法对比
![.png-2019-12-10-16-47-31](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/.png-2019-12-10-16-47-31)



