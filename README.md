

[TOC]



# Workspace of Conversation AI

# Target
+ Step 1. Collect current papers, corpus and projects
+ Step 2. Pipeline model
+ Step 3. End2End model

# Dataset

+ [chatbot corpus chinese](https://github.com/codemayq/chaotbot_corpus_Chinese)

+ [Dialog corpus](https://github.com/candlewill/Dialog_Corpus)

+ [chat_corpus](https://github.com/Marsan-Ma-zz/chat_corpus)

+ [bAbI](https://github.com/facebook/bAbI-tasks)

+ [dstc8-reddit-corpus](https://github.com/microsoft/dstc8-reddit-corpus)

+ ### DSTC

  - The Dialog State Tracking Challenge (DSTC) is an on-going series of research community challenge tasks. Each task released dialog data labeled with dialog state information, such as the user’s desired restaurant search query given all of the dialog history up to the current turn. The challenge is to create a “tracker” that can predict the dialog state for new dialogs. In each challenge, trackers are evaluated using held-out dialog data.

  #### DSTC1

  - DSTC1 used human-computer dialogs in the bus timetable domain. Results were presented in a special session at [SIGDIAL 2013](http://www.sigdial.org/workshops/sigdial2013/). DSTC1 was organized by Jason D. Williams, Alan Black, Deepak Ramachandran, Antoine Raux.
  - Data : https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/#!dstc1-downloads
  - Project:
    - pass
    - pass

  #### DSTC2 and DSTC3

  - DSTC2/3 used human-computer dialogs in the restaurant information domain. Results were presented in special sessions at [SIGDIAL 2014](http://www.sigdial.org/workshops/conference15/) and [IEEE SLT 2014](http://www.slt2014.org/). DSTC2 and 3 were organized by Matthew Henderson, Blaise Thomson, and Jason D. Williams.
  - Data : http://camdial.org/~mh521/dstc/
  - Project:
    - pass
    - pass

  #### DSTC4

  - DSTC4 used human-human dialogs in the tourist information domain. Results were presented at [IWSDS 2015](http://www.iwsds.org/). DSTC4 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, and Jason D. Williams.
  - Data:
    - http://www.colips.org/workshop/dstc4/data.html
  - Project:
    - pass

  #### DSTC5

  - DSTC5 used human-human dialogs in the tourist information domain, where training dialogs were provided in one language, and test dialogs were in a different language. Results were presented in a special session at [IEEE SLT 2016](http://www.slt2016.org/). DSTC5 was organized by Seokhwan Kim, Luis F. D’Haro, Rafael E Banchs, Matthew Henderson, Jason D. Williams, and Koichiro Yoshino.
  - Data:
    - http://workshop.colips.org/dstc5/data.html
  - Project:
    - Pass

  #### DSTC6

  - DSTC6 consisted of 3 parallel tracks:
    - End-to-End Goal Oriented Dialog Learning
    - End-to-End Conversation Modeling
    - Dialogue Breakdown Detection.
  - Results will be presented at a workshop immediately after NIPS 2017.
  - DSTC6 is organized by Chiori Hori, Julien Perez, Koichiro Yoshino, and Seokhwan Kim.
  - Tracks were organized by Y-Lan Boureau, Antoine Bordes, Julien Perez, Ryuichi Higashinaka, Chiori Hori, and Takaaki Hori.

  ### Ubuntu Dialogue Corpus

  - The Ubuntu Dialogue Corpus : A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems, 2015 [[paper\]](http://arxiv.org/abs/1506.08909) [[data\]](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

  ### Goal-Oriented Dialogue Corpus

  - **(Frames)** Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems, 2016 [[paper\]](https://arxiv.org/abs/1704.00057) [[data\]](http://datasets.maluuba.com/Frames)
  - **(DSTC 2 & 3)** Dialog State Tracking Challenge 2 & 3, 2013 [[paper\]](http://camdial.org/~mh521/dstc/downloads/handbook.pdf) [[data\]](http://camdial.org/~mh521/dstc/)

  ### Standford

  - A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset
    - Mihail Eric and Lakshmi Krishnan and Francois Charette and Christopher D. Manning. 2017. Key-Value Retrieval Networks for Task-Oriented Dialogue. In Proceedings of the Special Interest Group on Discourse and Dialogue (SIGDIAL). https://arxiv.org/abs/1705.05414. [pdf]
    - https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
    - http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip
    - calendar scheduling
    - weather information retrieval
    - point-of-interest navigation

  ### Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems

  - Maluuba 放出的对话数据集。
  - 论文链接：http://www.paperweekly.site/papers/407
  - 数据集链接：http://datasets.maluuba.com/Frames

  ### Multi WOZ

  - https://www.repository.cam.ac.uk/handle/1810/280608

  ### Stanford Multi-turn Multi-domain

  - 包含三个domain（日程，天气，景点信息），可参考下该数据机标注格式：
  - https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
  - 论文citation
  - Key-Value Retrieval Networks for Task-Oriented Dialogue https://arxiv.org/abs/1705.05414

  ### A Survey of Available Corpora for Building Data-Driven Dialogue Systems

  - 把所有的数据集按照不同类别进行分类总结，里面涵盖了很多数据集
  - [链接](http://link.zhihu.com/?target=https%3A//docs.google.com/spreadsheets/d/1SJ4XV6NIEl_ReF1odYBRXs0q6mTkedoygY3kLMPjcP8/pubhtml)

<<<<<<< HEAD
  ### 英文数据集
=======
#### DSTC7

#### DSTC8

### Ubuntu Dialogue Corpus
>>>>>>> 747066e768ff522b2c9dcee1a9aaaf3274806a8c

  - Cornell Movie Dialogs：电影对话数据集，下载地址：[http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html](http://link.zhihu.com/?target=http%3A//www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
  - Ubuntu Dialogue Corpus：Ubuntu日志对话数据，下载地址：[https://arxiv.org/abs/1506.08909](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/1506.08909)
  - OpenSubtitles：电影字幕，下载地址：[http://opus.lingfil.uu.se/OpenSubtitles.php](http://link.zhihu.com/?target=http%3A//opus.lingfil.uu.se/OpenSubtitles.php)
  - Twitter：twitter数据集，下载地址：[https://github.com/Marsan-Ma/twitter_scraper](http://link.zhihu.com/?target=https%3A//github.com/Marsan-Ma/twitter_scraper)
  - Papaya Conversational Data Set：基于Cornell、Reddit等数据集重新整理之后，好像挺干净的，下载链接：[https://github.com/bshao001/ChatLearner](http://link.zhihu.com/?target=https%3A//github.com/bshao001/ChatLearner)

  相关数据集的处理代码或者处理好的数据可以参见下面两个github项目：

  - [DeepQA](http://link.zhihu.com/?target=https%3A//github.com/Conchylicultor/DeepQA)
  - [chat_corpus](http://link.zhihu.com/?target=https%3A//github.com/Marsan-Ma/chat_corpus)
  - dgk_shooter_min.conv：中文电影台词数据集，下载链接：[https://github.com/rustch3n/dgk_lost_conv](http://link.zhihu.com/?target=https%3A//github.com/rustch3n/dgk_lost_conv)
  - 白鹭时代中文问答语料：白鹭时代论坛问答数据，一个问题对应一个最好的答案。下载链接：[https://github.com/Samurais/egret-wenda-corpus](http://link.zhihu.com/?target=https%3A//github.com/Samurais/egret-wenda-corpus)
    - 微博数据集：华为李航实验室发布，也是论文“Neural Responding Machine for Short-Text Conversation”使用的数据集下载链接：[http://61.93.89.94/Noah_NRM_Data/](http://link.zhihu.com/?target=http%3A//61.93.89.94/Noah_NRM_Data/)
  - 新浪微博数据集，评论回复短句，下载地址：[http://lwc.daanvanesch.nl/openaccess.php]
# Resource

+ pass



# Metric



# Solutions

## Chat-Bot

## IR-Bot

## QA-Bot

## Task-Bot

## Pipeline

### ASR

- APIs or Tools for free

### NLU

- Domain CLF
  - context based domain clf
- Intent Detection
- Slot Filling
- Joint Learning and Ranking

### DM

- DST
- DPL

### NLG

### TTS



# Problem

# Open Issues

# Milestone



# Coding Standards

# Usages

# Reference

## Links

- [AIComp Top](https://github.com/linxid/AICompTop)
- [Robot Design](https://github.com/qhduan/ConversationalRobotDesign)
- [sizhi bot](https://github.com/ownthink/robot]
- [home assistant](https://github.com/home-assistant/home-assistant)
- [textClassifier](https://github.com/jiangxinyang227/textClassifier)

## Projects

+ [2018 JDDC对话大赛亚军解决方案 Dialog-System-with-Task-Retrieval-and-Seq2seq](https://github.com/Dikea/Dialog-System-with-Task-Retrieval-and-Seq2seq)
+ [seq2seq chatbot](https://github.com/lc222/seq2seq_chatbot)
+ [Task-Oriented-Dialogue-Dataset-Survey](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey)
+ [jddc_solution_4th](https://github.com/zengbin93/jddc_solution_4th)
+ [jddc_baseline_tfidf](https://github.com/SimonJYang/JDDC-Baseline-TFIDF)
+ [jddc_baseline_seq2seq](https://github.com/SimonJYang/JDDC-Baseline-TFIDF)
+ [transformer_chatbot](https://github.com/atselousov/transformer_chatbot)
+ [Seq2Seq_Chatbot_QA](https://github.com/qhduan/Seq2Seq_Chatbot_QA)
+ [rasa_chatbot](https://github.com/zqhZY/_rasa_chatbot)
+ [TC-bot](https://github.com/MiuLab/TC-Bot)
+ [DNN-DST](https://github.com/CallumMain/DNN-DST)
+ [chatbot-MemN2N-tf](https://github.com/vyraun/chatbot-MemN2N-tensorflow)
+ [DST](https://github.com/voicy-ai/DialogStateTracking)

+ [LatticeLSTM](https://github.com/jiesutd/LatticeLSTM)
+ [Text CLF](https://github.com/jatana-research/Text-Classification)
+ [Text CLF cnn-rnn-tf](https://github.com/gaussic/text-classification-cnn-rnn)
+ [text_classification](https://github.com/brightmart/text_classification)

### Personalizing Dialogue Agents: I have a dog, do you have pets too?

- 本文是 Facebook AI Research 发表于 NIPS 2018 的工作。论文根据一个名为 PERSONA-CHAT 的对话数据集来训练基于 Profile 的聊天机器人，该数据集包含超过 16 万条对话。
- 本文致力于解决以下问题：
- 聊天机器人缺乏一致性格特征
  - 聊天机器人缺乏长期记忆
  - 聊天机器人经常给出模糊的回应，例如 I don't know
- 数据集链接
- https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat
