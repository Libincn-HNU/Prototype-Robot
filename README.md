---
noteId: "28910f1019bd11eabcd5bf51d3f5c99a"
tags: []

---

<!-- TOC -->

1. [Target](#target)
    1. [整理Bot 相关的数据，论文，例子，代码](#整理bot-相关的数据论文例子代码)
    2. [探索情感的构造，并依靠现有工具进行搭建](#探索情感的构造并依靠现有工具进行搭建)
    3. [探索逻辑的构造，并依靠现有工具进行搭建](#探索逻辑的构造并依靠现有工具进行搭建)
2. [Problem](#problem)
3. [Doing](#doing)
4. [ToDo List](#todo-list)

<!-- /TOC -->


# Target
## 整理Bot 相关的数据，论文，例子，代码
## 探索情感的构造，并依靠现有工具进行搭建
## 探索逻辑的构造，并依靠现有工具进行搭建
+ 信息完善
    + 多轮对话指代消解

# Problem

+ 意图判读
    + 介绍下香港
        + 百科 + 摘要
        + 实时爬取 + 摘要
    + 我想去香港
        + 任务型
    + 河南烩面
        + 可以选择 网上购买 河南烩面 周围的饭店 还是 介绍河南烩面 
        + 单独出现一个词条的
    + 你看过三体吗？
        + 不适合 回答 "没看过啊"
        + 直接回答 “消灭人类暴政，世界属于三体"也不合适
        + 比价好的单轮回答是，"看过啊，我还知道消灭人类暴政，世界属于三体"
        

+ 如何加入知识？
    + 加入各种资源库 KB 
    + 搭建问答模型
+ 多个模型融合排序，选择最佳模型

+ 基于IR 的 sp QA
    + more deep 


# Doing
+ seqGan 和其他 生成类方法
+ IR bot KB bot 等其他方法
+ Seq2Seq with embedding
+ 抽取式
+ 生成式
+ 知识库构建，查询，更新

# ToDo List
+ A New Archtechture for Multi-turn Response Selection in Retrieval-based Chatbots
+ seqGAN
    + exector.py 
        + merge_data_for_disc 方法中的 decoder 方法中 使用 gen_model.step， 连续执行几次step 结果均不变
        + 可能是 参数问题导致迭代的太慢，又有可能是代码逻辑问题

+ beam search， antilm， pointer network
