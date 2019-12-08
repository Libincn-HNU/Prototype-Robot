---
noteId: "2656b6d019ba11eabcd5bf51d3f5c99a"
tags: []

---

# Dataset 

## 中文数据集

语料名称 | 语料数量 | 语料来源说明 | 语料特点 | 语料样例 | 是否已分词
---|---|---|---|---|---
[chatterbot](https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese) | 560 | 开源项目 | 按类型分类，质量较高  | Q:你会开心的 A:幸福不是真正的可预测的情绪。 | 否
[douban 豆瓣多轮](https://github.com/MarkWuNLP/MultiTurnResponseSelection ) | 352W | 来自北航和微软的paper, 开源项目 | 噪音相对较少，原本是多轮（平均7.6轮）  | Q:烟台 十一 哪 好玩 A:哪 都 好玩 · · · · | 是
[ptt PTT八卦语料](https://github.com/zake7749/Gossiping-Chinese-Corpus) | 40W | 开源项目，台湾PTT论坛八卦版 | 繁体，语料较生活化，有噪音  | Q:为什么乡民总是欺负国高中生呢QQ	A:如果以为选好科系就会变成比尔盖兹那不如退学吧  | 否
qingyun（青云语料） | 10W | 某聊天机器人交流群 | 相对不错，生活化  | Q:看来你很爱钱 	 A:噢是吗？那么你也差不多了 | 否
[subtitle 电视剧对白语料](https://github.com/fateleak/dgk_lost_conv) | 274W | 开源项目，来自爬取的电影和美剧的字幕 | 有一些噪音，对白不一定是严谨的对话，原本是多轮（平均5.3轮）  | Q:京戏里头的人都是不自由的	A:他们让人拿笼子给套起来了了 | 否
tieba 贴吧论坛回帖语料 | 232W | 偶然找到的 | 多轮，有噪音 https://pan.baidu.com/s/1mUknfwy1nhSM7XzH8xi7gQ 密码:i4si  | Q:前排，鲁迷们都起床了吧	A:标题说助攻，但是看了那球，真是活生生的讽刺了 | 否
weibo（微博语料） | 443W | 华为 Noah 实验室 Neural Responding Machine for Short-Text Conversation | 仍有一些噪音  | Q:北京的小纯洁们，周日见。#硬汉摆拍清纯照# A:嗷嗷大湿的左手在干嘛，看着小纯洁撸么。 | 否
[xiaohuangji（小黄鸡语料）](https://github.com/candlewill/Dialog_Corpus) | 45W | 原人人网项目语料 | 有一些不雅对话，少量噪音 | Q:你谈过恋爱么	A:谈过，哎，别提了，伤心..。 | 否


## 中文其他数据集
- 三千万字幕语料
https://link.zhihu.com/?target=http%3A//www.shareditor.com/blogshow/%3FblogId%3D112

- 白鹭时代中文问答语料
    - 白鹭时代论坛问答数据，一个问题对应一个最好的答案。下载链接：https://github.com/Samurais/egret-wenda-corpus
- 微博数据集
    - 华为李航实验室发布，也是论文“Neural Responding Machine for Short-Text Conversation”使用的数据集下载链接：http://61.93.89.94/Noah_NRM_Data/
- 新浪微博数据集
    - 评论回复短句，下载地址：http://lwc.daanvanesch.nl/openaccess.php
    
## 英文其他数据集

### Cornell Movie Dialogs：电影对话数据集，下载地址：http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
### Ubuntu Dialogue Corpus：Ubuntu日志对话数据，下载地址：https://arxiv.org/abs/1506.08909
+ UDC 1.0 100W 多轮对话数据
### OpenSubtitles：电影字幕，下载地址：http://opus.lingfil.uu.se/OpenSubtitles.php
### Twitter：twitter数据集，下载地址：https://github.com/Marsan-Ma/twitter_scraper
### Papaya Conversational Data Set：基于Cornell、Reddit等数据集重新整理之后，好像挺干净的，下载链接：https://github.com/bshao001/ChatLearner

## JDDC

+ 需要注册才能得到数据集
+ 有待上传

## Person-Chat
    + Facebook
    + 16w 条

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

## Ubuntu Dialogue Corpus

- The Ubuntu Dialogue Corpus : A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems, 2015 [[paper\]](http://arxiv.org/abs/1506.08909) [[data\]](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
  
## Goal-Oriented Dialogue Corpus
  
- **(Frames)** Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems, 2016 [[paper\]](https://arxiv.org/abs/1704.00057) [[data\]](http://datasets.maluuba.com/Frames)
- **(DSTC 2 & 3)** Dialog State Tracking Challenge 2 & 3, 2013 [[paper\]](http://camdial.org/~mh521/dstc/downloads/handbook.pdf) [[data\]](http://camdial.org/~mh521/dstc/)
  
## Standford
  
- A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset
- Mihail Eric and Lakshmi Krishnan and Francois Charette and Christopher D. Manning. 2017. Key-Value Retrieval Networks for Task-Oriented Dialogue. In Proceedings of the Special Interest Group on Discourse and Dialogue (SIGDIAL). https://arxiv.org/abs/1705.05414. [pdf]
- https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
- http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip
  - calendar scheduling
- weather information retrieval
  - point-of-interest navigation
  
## Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems

- Maluuba 放出的对话数据集。
- 论文链接：http://www.paperweekly.site/papers/407
  - 数据集链接：http://datasets.maluuba.com/Frames

## Multi WOZ

- https://www.repository.cam.ac.uk/handle/1810/280608
  
## Stanford Multi-turn Multi-domain
  
- 包含三个domain（日程，天气，景点信息），可参考下该数据机标注格式：
  - https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
- 论文citation
  - Key-Value Retrieval Networks for Task-Oriented Dialogue https://arxiv.org/abs/1705.05414
  
- 把所有的数据集按照不同类别进行分类总结，里面涵盖了很多数据集