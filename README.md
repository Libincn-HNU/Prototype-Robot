<!-- TOC -->

- [Target](#target)
  - [整理Bot 相关的数据，论文，例子，代码](#%e6%95%b4%e7%90%86bot-%e7%9b%b8%e5%85%b3%e7%9a%84%e6%95%b0%e6%8d%ae%e8%ae%ba%e6%96%87%e4%be%8b%e5%ad%90%e4%bb%a3%e7%a0%81)
  - [探索情感的构造，并依靠现有工具进行搭建](#%e6%8e%a2%e7%b4%a2%e6%83%85%e6%84%9f%e7%9a%84%e6%9e%84%e9%80%a0%e5%b9%b6%e4%be%9d%e9%9d%a0%e7%8e%b0%e6%9c%89%e5%b7%a5%e5%85%b7%e8%bf%9b%e8%a1%8c%e6%90%ad%e5%bb%ba)
  - [探索逻辑的构造，并依靠现有工具进行搭建](#%e6%8e%a2%e7%b4%a2%e9%80%bb%e8%be%91%e7%9a%84%e6%9e%84%e9%80%a0%e5%b9%b6%e4%be%9d%e9%9d%a0%e7%8e%b0%e6%9c%89%e5%b7%a5%e5%85%b7%e8%bf%9b%e8%a1%8c%e6%90%ad%e5%bb%ba)
- [Category](#category)
  - [获取答案的方式](#%e8%8e%b7%e5%8f%96%e7%ad%94%e6%a1%88%e7%9a%84%e6%96%b9%e5%bc%8f)
  - [业务场景](#%e4%b8%9a%e5%8a%a1%e5%9c%ba%e6%99%af)
  - [答案类型](#%e7%ad%94%e6%a1%88%e7%b1%bb%e5%9e%8b)
- [Solutions](#solutions)
  - [Contextual](#contextual)
  - [DM&amp;UP](#dmampup)
  - [FQA](#fqa)
    - [KBQA](#kbqa)
    - [IRQA](#irqa)
    - [Community QA](#community-qa)
    - [Open Domain QA](#open-domain-qa)
    - [MRC](#mrc)
- [Problem](#problem)
- [Doing](#doing)
- [ToDo](#todo)
- [Backlog](#backlog)

<!-- /TOC -->
# Target
## 整理Bot 相关的数据，论文，例子，代码
## 探索情感的构造，并依靠现有工具进行搭建
## 探索逻辑的构造，并依靠现有工具进行搭建
+ 信息完善
    + 多轮对话指代消解

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
