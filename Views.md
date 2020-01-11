


# Links
- ACL 2019
    - http://www.sohu.com/a/333666460_129720
    - 包括 Neural Conversation(已经整理)， Task Oriented， New Task 三部分
- QA 对解析的七种方法和优化思路
    - https://www.jianshu.com/p/583ae40e93cd
- deep attention match 多轮对话跟踪模型
    - https://zhuanlan.zhihu.com/p/82004144
- MatchZoo : A Toolkit for deep text matching
- [AIComp Top](https://github.com/linxid/AICompTop)
- [Robot Design](https://github.com/qhduan/ConversationalRobotDesign)
- [sizhi bot](https://github.com/ownthink/robot]
- [home assistant](https://github.com/home-assistant/home-assistant)
- 评价指标
    - https://blog.csdn.net/liuchonge/article/details/79104045
- GAN + 文本生成的一些论文
- DPF 无监督对话数据清洗方法
    - http://breezedeus.github.io/2017/08/07/breezedeus-purify-data.html

+ A Survey of Available Corpora for Building Data-Driven Dialogue Systems
    - (https://arxiv.org/pdf/1512.05742.pdf

+ A Neural Conversation Model/
    + https://arxiv.org/abs/1506.05869

+ Neural Response Generation via GAN with an APProximate Embedding Layer
    + 单轮回答，抑制安全回答
    + 实现  
    + https://github.com/lan2720/GAN-AEL
    + https://github.com/deepanshugarg257/Response-Generation-with-AEL

+ Deep Reinforcement Learning for Dialogue Generation
    + https://www.cnblogs.com/jiangxinyang/p/10469860.html

    + 传统的seq2seq 问题
    + 1> 安全回答
    + 2> 使用MLE 容易死循环

    + 首先使用Seq-to-Seq模型预训练一个基础模型，然后根据作者提出的三种Reward来计算每次生成的对话的好坏，并使用policy network的方法提升对话响应的多样性、连贯性和对话轮次。文章最大的亮点就在于定义了三种reward（Ease of answering、Information Flow、Semantic Coherence），分别用于解决dull response、repetitive response、ungrammatical response
    + https://zhuanlan.zhihu.com/p/21587758


## Projects

### JDDC
+ [2018 JDDC对话大赛亚军解决方案 Dialog-System-with-Task-Retrieval-and-Seq2seq](https://github.com/Dikea/Dialog-System-with-Task-Retrieval-and-Seq2seq)
+ [jddc_solution_4th](https://github.com/zengbin93/jddc_solution_4th)
+ [seq2seq chatbot](https://github.com/lc222/seq2seq_chatbot)
  + 使用 seq2seq 构建简单系统
  + embedding、attention、beam_search等功能，数据集是Cornell Movie Dialogs
+ [jddc_baseline_tfidf](https://github.com/SimonJYang/JDDC-Baseline-TFIDF)
  + 首先计算用户的问题与问题库中的问题的相似度并选出top15的相似问题，然后去问题库对应的答案库中找出这15个问题对应的答案， 以此作为回答用户问题的候选答案

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

### Others
+ [TC-bot](https://github.com/MiuLab/TC-Bot)




# Papers

## AAAI 2020

### Generative Adversarial Zero-shot Relation Learning for Knowledge Graphs
+ 知识图谱的生成式对抗零样本关系学习

## NIPS 2019

## EMNLP 2019

### Knowledge Aware Conversation Generation with Explainable Reasoing ever Augmented Graphs
+ Baidu
+ Tips 
  + 大部分模型 容易出现安全回复和不连贯回复，这是因为仅仅从语料中学习语义而不是借助背景知识
  + 引入结构化信息
    + 利用三元组或者图路径来缩小知识的候选范围并增强模型的泛化能力
    + 但是选出的知识往往是实体或普通词，因而无法为回复生成更加丰富的信息
  + 引入非结构化信息
    + 文本知识(电影评论或者电影剧情)可以为回复的生成提供丰富的参考信息，但是非结构化的表示方案要求模型具有很强的能力来从知识文本集合中进行知识选择或者使用注意力机制
+ 此文综合使用结构化信息和非结构化信息， 提出了基于扩充知识图(Augmented Knowledge Graph)开放域对话生成模型，模型由知识选择和回复生成这两个模块组成

### A Discrete CVAE for Response Generation on Short-Text Conversation

### A Descrete Hard EM Approach for Weakly Supervised Question Answering

###  A Logic-Driven Framework for consistency of Neural Models

###  A Multi-type Multi-span Network for Reading Comprehension that requires discrete readsoning

### A Practical Dialogue-Act-Driven Conversation Model for Multi-Ture Response Selection

### A Semi-supervised stable variational network for promoting replier-consistency in dialogue generation

### Adaptive Parameterization for Neural Dialogue Generation

### Addressing Semantic Drift in Question Generation for Semi-Supervised Question Answering

### Adversarial Domain Adaptation for Machine Reading Comprehension

### An End-to-End Generative Architecture for Paraphrase Generation

### Answer-guided and Semantic Coherent Question Generation in Open-domain Conversation

### Answering Complex Open-domain Questions Through Iterative Query Generation

### Answering questions by learning to rank - Learning to rank by answering questions

### ARAML: A Stable Adversarial Training Framework for Text Generation

### Are You for Real? Detecting Identity Fraud via Dialogue Interactions

### Asking Clarification Questions in Knowledge-Based Question Answering

### Attending to Future Tokens for Bidirectional Sequence Generation

### Data-Efficient Goal-Oriented Conversation with Dialogue Knowledge Transfer Networks

### Deep Copycat Networks for Text-to-Text Generation

### Denoising-based Sequence-to-Sequence Pre-training for Text Generation

### Dialog Intent Induction with Deep Multi-View Clustering

### DialogueGCN: A Graph-based Network for Emotion Recognition in Conversation

### Discourse-Aware Semantic Self-Attention for Narrative Reading Comprehension

### DyKgChat: Benchmarking Dialogue Generation Grounding on Dynamic Knowledge Graphs

### Entity-Consistent End-to-end Task-Oriented Dialogue System with KB Retriever

### GECOR: An End-to-End Generative Ellipsis and Co-reference Resolution Model for Task-Oriented Dialogue

### Generating Questions for Knowledge Bases via Incorporating Diversified Contexts and Answer-Aware Loss

### Guided Dialog Policy Learning: Reward Estimation for Multi-Domain Task-Oriented Dialog
+ DPL

### Hierarchy Response Learning for Neural Conversation Generation

### How to Build User Simulators to Train RL-based Dialog Systems
+ RL

### Improving Open-Domain Dialogue Systems via Multi-Turn Incomplete Utterance Restoration

### Incorporating External Knowledge into Machine Reading for Generative Question Answering

### KagNet: Learning to Answer Commonsense Questions with Knowledge-Aware Graph Networks

### Task-Oriented Conversation Generation Using Heterogeneous Memory Networks
+ 结合context和knowledge base进行回答
+ 小蜜团队EMNLP 2019上的文章，本文提出了一种异构记忆网络（Heterogeneous Memory Networks, HMNs），将动态的context和静态的knowledge base分别用不同的记忆网络存储联合建模来解决这类问题。
+ 论文链接：https://www.aclweb.org/anthology/D19-1463.pdf 

### Knowledge-Enriched Transformer for Emotion Detection in Textual Conversations

### KnowledgeNet: A Benchmark Dataset for Knowledge Base Population

### Learning to Update Knowledge Graph by Reading News
+ 更新图谱

### Learning with Limited Data for Multilingual Reading Comprehension

### Meta Relational Learning for Few-Shot Link Prediction in Knowledge Graphs

### Model-based Interactive Semantic Parsing: A Unified Formulation and A Text-to-SQL Case Study
+ text-to-sql

### Multi-hop Selector Network for Multi-turn Response Selection in Retrieval-based Chatbots

### Multi-task Learning for Conversational Question Answering Over a Large-Scale Knowledge Base
+ MultiDoGO: Multi-Domain Goal-Oriented Dialogues

### NumNet: Machine Reading Comprehension with Numerical Reasoning
+ Numerical Reasoning

### PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text

### Ranking and Sampling in Open-domain Question Answering

### Task-Oriented Conversation Generation Using Heterogeneous Memory Networks

### TaskMaster Dialog Corpus: Toward a Realistic and Diverse Dataset

### Towards Controllable and Personalized Review Generation

### Towards Knowledge-Based Recommender Dialog System

### Variational Hierarchical User-based Conversation Model

### What’s Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering

### Who Is Speaking to Whom? Learning to Identify Utterance Addressee in Multi-Party Conversations

## ACL2019
+ Dialogue and Interactive Systems 主题 长论文 38 篇，分为Neural Conversation Model， Task Oriented Dialogue 和 New Task
+ Neural Conversation Model
    + $P_\theta(y|x) = \pi_{i=1}^{n} P_\theta(y_i|y_{<i}, x_1,x_2,...,x_n)$

### Boosting dialog response generation
+ 优化 通用回答和安全回答的问题 
+ 使用 RAML(Reward-augmented Maximum likelihood learning) 

### Do Neural Dialog Systems Use the Conversation History Effectively? An Empirical Study
+ Bengio
+ 研究现有神经网络模型，能否有效利用历史
    + 预测阶段加入扰动，但是结果基本没有变化

### Constructing Interpretive Spatio-Temporal Features for Multi-Turn Response Selection
+ 本文通过加入时序和空间上的feature，来解决对话系统中回复句子的选择问题
+ 首先通过**软对齐**来获取上下文和回复之间的关联信息
+ 其次是时间维度聚合注意力镜像，利用3D卷积和池化来抽取匹配信息

### Improving Multi-turn Dialogue Modelling with Utterence Rewriter
+ 腾讯 阿里
+ 通过语句改写来解决多轮对话中信息省略和引用的问题
+ 构建数据集
+ 使用encoder decoder 来进行指代消解

### Incremental Transformer with Deliberation Decoder for Document Grounded Conversations
+ 提出在有文档信息的多轮对话中，一种基于transformer的对话生成模型
+ 模型的输入包括 对话历史 和 多轮对话的任务一个相关的文档(如何判断相关)
    + 挖掘文档中和对话相关的部分
    + 将多轮对话的语句 和文档中相关的部分进行统一的表示
+ 架构
    + 增量式transformer，将对话的句子以及相关联的文档增加到模型中
    + 两阶段解码，第一阶段关注对话上下文的连贯性，第二阶段引入相关的文档内容，来对第一阶段的文档进行润饰

### One Time of Interection May Not Be Enough : Go Deep with an Interaction-over-Interaction Network for Response Selection in Dialogues
+ 提出一个基于检索的深度交互对话模型，来解决现有模型中，对对话交互信息利用较浅的问题
+ 定义Interaction-over-Interaction 网络
    + 自注意力模块：建模QQ，QA，AA 之间的依赖
    + 交互模块：交互建模
    + 压缩模块：结果合并

### Text Geneartion from Knowledge Graphs with Graph Transformers
+ ACL 2019
+ https://blog.csdn.net/TgqDT3gGaMdkHasLZv/article/details/100190240

+ 提出了一种Graph Transformer编码方法用于知识图谱表示学习
+ 提出一种将IE输出转换为图结构用于编码的过程
+ 构建了一个可复用的大型“图谱-文本”对数据集

### Vocabulary Pyramid Network: Multi-Pass Encoding and Decoding with Multi-Level Vocabularies for Response Generation

+ seq2seq框架作为文本生成的主流框架，在对话领域已被广泛应用。然而在对话生成时，seq2seq倾向于生成通用的答案，缺乏进一步的润色修饰；***解码的过程中词表大小的限制与逐词解码的方式存在偏置问题***，***且目标端的全局信息无法利用***。如何在解码过程中不断丰富词表，利用全局信息丰富句子内容是本论文的主要研究贡献。
+ 论文链接：https://www.aclweb.org/anthology/P19-1367/ 

## NIPS 2018
### Personalizing Dialogue Agents: I have a dog, do you have pets too?
- https://www.jianshu.com/p/189cba8f2069

- 本文是 Facebook AI Research 发表于 NIPS 2018 的工作。论文根据一个名为 PERSONA-CHAT 的对话数据集来训练基于 Profile 的聊天机器人，该数据集包含超过 16 万条对话。
- 本文致力于解决以下问题：
- 聊天机器人缺乏一致性格特征
  - 聊天机器人缺乏长期记忆
  - 聊天机器人经常给出模糊的回应，例如 I don't know
  
- 数据集链接
- https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat

- Related Work 梳理的很好
