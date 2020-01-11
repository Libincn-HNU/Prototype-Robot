<!-- TOC -->

1. [Target](#target)
    1. [整理Bot 相关的数据，论文，例子，代码](#整理bot-相关的数据论文例子代码)
    2. [探索情感的构造，并依靠现有工具进行搭建](#探索情感的构造并依靠现有工具进行搭建)
    3. [探索逻辑的构造，并依靠现有工具进行搭建](#探索逻辑的构造并依靠现有工具进行搭建)
2. [Current and Deadline](#current-and-deadline)
3. [Category](#category)
    1. [获取答案的方式](#获取答案的方式)
    2. [业务场景](#业务场景)
    3. [答案类型](#答案类型)
4. [Solutions](#solutions)
    1. [Contextual](#contextual)
    2. [DM&UP](#dmup)
    3. [FQA](#fqa)
        1. [KBQA](#kbqa)
        2. [IRQA](#irqa)
        3. [Community QA](#community-qa)
        4. [Open Domain QA](#open-domain-qa)
        5. [MRC](#mrc)
5. [Problem](#problem)
6. [Doing](#doing)
7. [ToDo](#todo)
8. [Backlog](#backlog)
9. [Tooks](#tooks)
10. [Products](#products)

<!-- /TOC -->
# Target
## 整理Bot 相关的数据，论文，例子，代码
## 探索情感的构造，并依靠现有工具进行搭建
## 探索逻辑的构造，并依靠现有工具进行搭建
+ 信息完善
    + 多轮对话指代消解
    
# Current and Deadline
+ 单多轮数据预处理脚本
+ 单多轮检索调试()

# Category
## 获取答案的方式
+ 生成式
+ 检索式

## 业务场景
+ 闲聊
+ 任务型
+ 知识型

## 答案类型
+ 事实型
+ 列举型
+ 定义型
+ 交互型
  + 单轮
  + 多轮

# Solutions
## Contextual
+ 进行上下文级别的建模，综合应用词法分析，句法分析，篇章分析进行信息表示，信息抽取，意图识别，领域迁移

## DM&UP
+ 表示对话状态
+ 进行用户建模
+ 对bot 进行对话指导（任务型澄清，用户建模澄清）

## FQA
### KBQA

### IRQA

### Community QA

### Open Domain QA
+ 开放领域问答

### MRC
+ 机器阅读理解

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

# ToDo
+ 知识库构建，查询，更新

# Backlog
+ A New Archtechture for Multi-turn Response Selection in Retrieval-based Chatbots
+ seqGAN
    + exector.py 
        + merge_data_for_disc 方法中的 decoder 方法中 使用 gen_model.step， 连续执行几次step 结果均不变
        + 可能是 参数问题导致迭代的太慢，又有可能是代码逻辑问题

+ beam search， antilm， pointer network

# Tooks
+ ES
+ Rasa

# Products
+ Alexa Echo
+ 小爱同学 API
+ 小冰
    + http://breezedeus.github.io/2019/02/23/breezedeus-xiaoice-framework.html
    
+ 百度 Unit
    + https://ai.baidu.com/unit/v2#/sceneliblist

