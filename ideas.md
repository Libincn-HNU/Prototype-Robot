<!-- TOC -->

1. [Backlog](#backlog)
2. [Products](#products)
3. [Tools](#tools)
4. [Views](#views)

<!-- /TOC -->

# Backlog
+ A New Archtechture for Multi-turn Response Selection in Retrieval-based Chatbots
+ seqGAN
    + exector.py 
        + merge_data_for_disc 方法中的 decoder 方法中 使用 gen_model.step， 连续执行几次step 结果均不变
        + 可能是 参数问题导致迭代的太慢，又有可能是代码逻辑问题

+ beam search， antilm， pointer network

# Products
+ Alexa Echo
+ 小爱同学 API
+ 小冰
    + http://breezedeus.github.io/2019/02/23/breezedeus-xiaoice-framework.html
    
+ 百度 Unit
    + https://ai.baidu.com/unit/v2#/sceneliblist
    
# Tools
+ MatchZoo : A Toolkit for deep text matching
+ ES
+ Rasa

# Views
+ ACL 2019 http://www.sohu.com/a/333666460_129720
    + 包括 Neural Conversation(已经整理)， Task Oriented， New Task 三部分

+ QA 对解析的七种方法和优化思路
    + https://www.jianshu.com/p/583ae40e93cd
+ deep attention match 多轮对话跟踪模型
    + https://zhuanlan.zhihu.com/p/82004144

+ AIComp Top
+ Robot Design
    + [sizhi bot](https://github.com/ownthink/robot]
+ home assistant
+ 评价指标
    + https://blog.csdn.net/liuchonge/article/details/79104045
+ GAN + 文本生成的一些论文
+ DPF 无监督对话数据清洗方法
    + http://breezedeus.github.io/2017/08/07/breezedeus-purify-data.html
+ A Survey of Available Corpora for Building Data-Driven Dialogue Systems
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

+ Awesome-chatbot
+ rasa
    + Rasa
    + rasa_chatbot_cn
    + _rasa_chatbot
    + rasa_chatbot
+ dst
    + DST
    + DNN-DST
    + DST