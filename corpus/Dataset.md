<!-- TOC -->

1. [Dataset](#dataset)
    1. [中文数据集](#中文数据集)
        1. [chatterbot](#chatterbot)
        2. [豆瓣多轮](#豆瓣多轮)
        3. [PTT八卦语料](#ptt八卦语料)
        4. [青云语料](#青云语料)
        5. [subtitle 电视剧对白语料](#subtitle-电视剧对白语料)
        6. [贴吧论坛回帖语料](#贴吧论坛回帖语料)
        7. [微博语料](#微博语料)
        8. [新浪微博数据集](#新浪微博数据集)
        9. [小黄鸡语料](#小黄鸡语料)
        10. [中文电影对白语料](#中文电影对白语料)
        11. [The NUS SMS Corpus](#the-nus-sms-corpus)
        12. [Datasets for Natural Language Processing](#datasets-for-natural-language-processing)
        13. [白鹭时代中文问答语料](#白鹭时代中文问答语料)
        14. [Chat corpus repository](#chat-corpus-repository)
        15. [保险行业QA语料库](#保险行业qa语料库)
        16. [三千万字幕语料](#三千万字幕语料)
        17. [白鹭时代中文问答语料](#白鹭时代中文问答语料-1)
        18. [JDDC](#jddc)
    2. [英文其他数据集](#英文其他数据集)
        1. [Cornell Movie Dialogs](#cornell-movie-dialogs)
        2. [OpenSubtitles](#opensubtitles)
        3. [Twitter](#twitter)
        4. [Papaya Conversational Data Set](#papaya-conversational-data-set)
        5. [Person-Chat](#person-chat)
        6. [Ubuntu Dialogue Corpus](#ubuntu-dialogue-corpus)
        7. [Standford](#standford)
        8. [Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems](#frames-a-corpus-for-adding-memory-to-goal-oriented-dialogue-systems)
        9. [Multi WOZ](#multi-woz)
    3. [Goal-Oriented Dialogue Corpus](#goal-oriented-dialogue-corpus)
        1. [**(Frames)** Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems, 2016](#frames-frames-a-corpus-for-adding-memory-to-goal-oriented-dialogue-systems-2016)
        2. [**(DSTC 2 & 3)** Dialog State Tracking Challenge 2 & 3, 2013](#dstc-2--3-dialog-state-tracking-challenge-2--3-2013)
    4. [DSTC](#dstc)
        1. [DSTC1](#dstc1)
        2. [DSTC2 and DSTC3](#dstc2-and-dstc3)
        3. [DSTC4](#dstc4)
        4. [DSTC5](#dstc5)
        5. [DSTC6](#dstc6)
        6. [DSTC7](#dstc7)
        7. [DSTC8](#dstc8)
2. [Reference](#reference)

<!-- /TOC -->

# Dataset 

## 中文数据集
### chatterbot
+ https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese
+ 数据大小 : 560, 数量量很小，无法使用
+ 来源 : 开源项目 
+ 特点 : 按类型分类，质量较高
+ 未分词
+ 样例 : 
    + Q:你会开心的 
    + A:幸福不是真正的可预测的情绪

### 豆瓣多轮
+ https://github.com/MarkWuNLP/MultiTurnResponseSelection
+ 大小 : 352W 
+ 来源 : 
    + 来自北航和微软的paper, 开源项目
    + A New Archtechture for Multi-turn Response Selection in Retrieval-based Chatbots.
    + https://arxiv.org/pdf/1612.01627.pdf
+ 特点 : 噪音相对较少，原本是多轮（平均7.6轮）  
+ 已经分词
+ 样例
    + Q:烟台 十一 哪 好玩 
    + A:哪 都 好玩 · · · · 

### PTT八卦语料
+ https://github.com/zake7749/Gossiping-Chinese-Corpus
+ 数据大小 : 40W
+ 来源 : 开源项目，台湾PTT论坛八卦版 
+ 特点 : 繁体，语料较生活化，有噪音 
+ 未分词
+ 样例
    + Q:为什么乡民总是欺负国高中生呢QQ	
    + A:如果以为选好科系就会变成比尔盖兹那不如退学吧

### 青云语料
+ 数据大小 : 10W 
+ 来源 : 某聊天机器人交流群 
+ 特点 : 相对不错，生活化
+ 未分词
+ 样例 
    + Q:看来你很爱钱 	 
    + A:噢是吗？那么你也差不多了

### subtitle 电视剧对白语料
+ https://github.com/fateleak/dgk_lost_conv
+ 数据大小 : 274W 
+ 来源 : 开源项目，来自爬取的电影和美剧的字幕 
+ 特点 : 有一些噪音，对白不一定是严谨的对话，原本是多轮（平均5.3轮）  
+ 未分词
+ 样例
    + Q:京戏里头的人都是不自由的	
    + A:他们让人拿笼子给套起来了了

### 贴吧论坛回帖语料 
+ https://pan.baidu.com/s/1mUknfwy1nhSM7XzH8xi7gQ 密码:i4si  
+ 数据大小 : 232W 
+ 特点 : 多轮，有噪音 
+ 未分词
+ 样例 
    + Q:前排，鲁迷们都起床了吧	
    + A:标题说助攻，但是看了那球，真是活生生的讽刺了

### 微博语料
+ 华为李航实验室发布，也是论文“Neural Responding Machine for Short-Text Conversation”使用的数据集下载链接：http://61.93.89.94/Noah_NRM_Data/， 此链接似乎已经失效
+ 数据大小 : 443W 
+ 来源 : 华为 Noah 实验室 Neural Responding Machine for Short-Text Conversation 
+ 特点 : 仍有一些噪音  
+ 未分词
+ 样例 
    + Q:北京的小纯洁们，周日见。#硬汉摆拍清纯照# 
    + A:嗷嗷大湿的左手在干嘛，看着小纯洁撸么。

### 新浪微博数据集
- 评论回复短句
- 下载地址：http://lwc.daanvanesch.nl/openaccess.php

### 小黄鸡语料
+ https://github.com/candlewill/Dialog_Corpus
+ 数据大小 : 45W 
+ 来源 : 原人人网项目语料
+ 特点 : 有一些不雅对话，少量噪音 
+ 未分词
+ 样例
    + Q:你谈过恋爱么	
    + A:谈过，哎，别提了，伤心..。 
    
    dgk_shooter_min.conv.zip
    
### 中文电影对白语料
+ https://github.com/candlewill/Dialog_Corpus
+ 噪音比较大，许多对白问答关系没有对应好

### The NUS SMS Corpus
+ https://github.com/candlewill/Dialog_Corpus
+ 包含中文和英文短信息语料，据说是世界最大公开的短消息语料

###  Datasets for Natural Language Processing
+ https://github.com/candlewill/Dialog_Corpus
+ 这是他人收集的自然语言处理相关数据集，主要包含Question Answering，Dialogue Systems， Goal-Oriented Dialogue Systems三部分，都是英文文本。可以使用机器翻译为中文，供中文对话使用

### 白鹭时代中文问答语料
+ https://github.com/candlewill/Dialog_Corpus
+ 由白鹭时代官方论坛问答板块10,000+ 问题中，选择被标注了“最佳答案”的纪录汇总而成。人工review raw data，给每一个问题，一个可以接受的答案。目前，语料库只包含2907个问答。(备份)

### Chat corpus repository
+ https://github.com/candlewill/Dialog_Corpus
+ chat corpus collection from various open sources
+ 包括：开放字幕、英文电影字幕、中文歌词、英文推文

### 保险行业QA语料库
+ https://github.com/candlewill/Dialog_Corpus
+ 通过翻译 insuranceQA产生的数据集。train_data含有问题12,889条，数据 141779条，正例：负例 = 1:10； test_data含有问题2,000条，数据 22000条，正例：负例 = 1:10；valid_data含有问题2,000条，数据 22000条，正例：负例 = 1:10


### 三千万字幕语料
+ https://link.zhihu.com/?target=http%3A//www.shareditor.com/blogshow/%3FblogId%3D112

### 白鹭时代中文问答语料
- 白鹭时代论坛问答数据，一个问题对应一个最好的答案
- 下载链接：https://github.com/Samurais/egret-wenda-corpus

### JDDC
+ 需要注册才能得到数据集
+ 有待上传

    
## 英文其他数据集

### Cornell Movie Dialogs
+ 电影对话数据集
+ 下载地址：http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

### OpenSubtitles
+ 电影字幕
+ 下载地址：http://opus.lingfil.uu.se/OpenSubtitles.php

### Twitter
+ twitter数据集
+ 下载地址：https://github.com/Marsan-Ma/twitter_scraper

### Papaya Conversational Data Set
+ 基于Cornell、Reddit等数据集重新整理之后，好像挺干净的
+ 下载链接：https://github.com/bshao001/ChatLearner

### Person-Chat
    + Facebook
    + 16w 条
    
### Ubuntu Dialogue Corpus
- The Ubuntu Dialogue Corpus : A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems, 2015 
+ paper http://arxiv.org/abs/1506.08909
+ data https://github.com/rkadlec/ubuntu-ranking-dataset-creator

### Standford
- A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset
- Mihail Eric and Lakshmi Krishnan and Francois Charette and Christopher D. Manning. 2017. Key-Value Retrieval Networks for Task-Oriented Dialogue. In Proceedings of the Special Interest Group on Discourse and Dialogue (SIGDIAL). https://arxiv.org/abs/1705.05414. [pdf]
- https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
- http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip
  - calendar scheduling
- weather information retrieval
  - point-of-interest navigation
- 包含三个domain（日程，天气，景点信息），可参考下该数据机标注格式：
  - https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
- 论文citation
  - Key-Value Retrieval Networks for Task-Oriented Dialogue https://arxiv.org/abs/1705.05414
  
- 把所有的数据集按照不同类别进行分类总结，里面涵盖了很多数据集
  
### Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems
- Maluuba 放出的对话数据集。
- 论文链接：http://www.paperweekly.site/papers/407
  - 数据集链接：http://datasets.maluuba.com/Frames

### Multi WOZ
- https://www.repository.cam.ac.uk/handle/1810/280608
  
## Goal-Oriented Dialogue Corpus
  
### **(Frames)** Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems, 2016 
+ paper : https://arxiv.org/abs/1704.00057
+ data : http://datasets.maluuba.com/Frames
### **(DSTC 2 & 3)** Dialog State Tracking Challenge 2 & 3, 2013 
+ paper : http://camdial.org/~mh521/dstc/downloads/handbook.pdf
+ data : http://camdial.org/~mh521/dstc/
  
## DSTC

  - The Dialog State Tracking Challenge (DSTC) is an on-going series of research community challenge tasks. Each task released dialog data labeled with dialog state information, such as the user’s desired restaurant search query given all of the dialog history up to the current turn. The challenge is to create a “tracker” that can predict the dialog state for new dialogs. In each challenge, trackers are evaluated using held-out dialog data.

### DSTC1

  - DSTC1 used human-computer dialogs in the bus timetable domain. Results were presented in a special session at [SIGDIAL 2013](http://www.sigdial.org/workshops/sigdial2013/). DSTC1 was organized by Jason D. Williams, Alan Black, Deepak Ramachandran, Antoine Raux.
  - Data : https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/#!dstc1-downloads
  - Project:
    - pass
    - pass

### DSTC2 and DSTC3

- DSTC2/3 used human-computer dialogs in the restaurant information domain. Results were presented in special sessions at [SIGDIAL 2014](http://www.sigdial.org/workshops/conference15/) and [IEEE SLT 2014](http://www.slt2014.org/). DSTC2 and 3 were organized by Matthew Henderson, Blaise Thomson, and Jason D. Williams.
- Data : http://camdial.org/~mh521/dstc/
- Project:
  - pass
  - pass

### DSTC4

- DSTC4 used human-human dialogs in the tourist information domain. Results were presented at [IWSDS 2015](http://www.iwsds.org/). DSTC4 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, and Jason D. Williams.
- Data:
  - http://www.colips.org/workshop/dstc4/data.html
- Project:
  - pass

### DSTC5

- DSTC5 used human-human dialogs in the tourist information domain, where training dialogs were provided in one language, and test dialogs were in a different language. Results were presented in a special session at [IEEE SLT 2016](http://www.slt2016.org/). DSTC5 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, Jason D. Williams, and Koichiro Yoshino.
- Data:
  - http://workshop.colips.org/dstc5/data.html
- Project:
  - Pass

### DSTC6

- DSTC6 consisted of 3 parallel tracks:
  - End-to-End Goal Oriented Dialog Learning
  - End-to-End Conversation Modeling
  - Dialogue Breakdown Detection.
- Results will be presented at a workshop immediately after NIPS 2017.
  - DSTC6 is organized by Chiori Hori, Julien Perez, Koichiro Yoshino, and Seokhwan Kim.
- Tracks were organized by Y-Lan Boureau, Antoine Bordes, Julien Perez, Ryuichi Higashinaka, Chiori Hori, and Takaaki Hori.

### DSTC7

### DSTC8


# Reference

+ https://github.com/candlewill/Dialog_Corpus

+ https://github.com/codemayq/chinese_chatbot_corpus
    + 常见的中文语料，并附带网盘链接和预处理脚本
    + https://pan.baidu.com/s/1szmNZQrwh9y994uO8DFL_A 提取码：f2ex
    + 已经存储到百度云盘 /闲聊语料/raw_chat_corpus.zip

+ 单轮数据
    + https://chatbot-dataset.oss-cn-beijing.aliyuncs.com/single_round.zip

+ 多轮数据
    + https://chatbot-dataset.oss-cn-beijing.aliyuncs.com/multi_round.zip