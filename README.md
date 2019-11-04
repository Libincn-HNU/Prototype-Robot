<!-- TOC -->

- [1. Workspace of Conversation AI](#1-workspace-of-conversation-ai)
- [2. Target](#2-target)
- [3. Dataset](#3-dataset)
    - [3.1. DSTC](#31-dstc)
        - [3.1.1. DSTC1](#311-dstc1)
- [4. Resource](#4-resource)
- [5. Metric](#5-metric)
- [6. Solutions](#6-solutions)
    - [6.1. Chat-Bot](#61-chat-bot)
    - [6.2. IR-Bot](#62-ir-bot)
    - [6.3. QA-Bot](#63-qa-bot)
    - [6.4. KB-Bot](#64-kb-bot)
    - [6.5. Task-Bot](#65-task-bot)
    - [6.6. Pipeline](#66-pipeline)
        - [6.6.1. ASR](#661-asr)
        - [6.6.2. NLU](#662-nlu)
        - [6.6.3. DM](#663-dm)
        - [6.6.4. NLG](#664-nlg)
        - [6.6.5. TTS](#665-tts)
- [7. Problem](#7-problem)
- [8. Open Issues](#8-open-issues)
- [9. Milestone](#9-milestone)
- [10. Coding Standards](#10-coding-standards)
- [11. Usages](#11-usages)
- [12. Reference](#12-reference)
    - [12.1. Links](#121-links)
    - [12.2. Papers](#122-papers)
        - [12.2.1. Knowledge Aware Conversation Generation with Explainable Reasoing ever Augmented Graphs](#1221-knowledge-aware-conversation-generation-with-explainable-reasoing-ever-augmented-graphs)
    - [12.3. Projects](#123-projects)
        - [12.3.1. Personalizing Dialogue Agents: I have a dog, do you have pets too?](#1231-personalizing-dialogue-agents-i-have-a-dog-do-you-have-pets-too)
        - [12.3.2. Seq2seq code build](#1232-seq2seq-code-build)
        - [12.3.3. DeepQA](#1233-deepqa)
        - [12.3.4. RasaHQ](#1234-rasahq)
        - [12.3.5. NLG/GAN](#1235-nlggan)
        - [12.3.6. Dual Training](#1236-dual-training)
        - [12.3.7. More Deep](#1237-more-deep)
        - [12.3.8. A Neural Conversation Model](#1238-a-neural-conversation-model)

<!-- /TOC -->


# Workspace of Conversation AI
<a id="markdown-workspace-of-conversation-ai" name="workspace-of-conversation-ai"></a>

# Target
<a id="markdown-target" name="target"></a>
+ Step 1. Collect current papers, corpus and projects
+ Step 2. Pipeline model
+ Step 3. End2End model

# Dataset
<a id="markdown-dataset" name="dataset"></a>

+ [chatbot corpus chinese](https://github.com/codemayq/chaotbot_corpus_Chinese)

+ [Dialog corpus](https://github.com/candlewill/Dialog_Corpus)

+ [chat_corpus](https://github.com/Marsan-Ma-zz/chat_corpus)

+ [bAbI](https://github.com/facebook/bAbI-tasks)

+ [dstc8-reddit-corpus](https://github.com/microsoft/dstc8-reddit-corpus)

## DSTC
<a id="markdown-dstc" name="dstc"></a>

  - The Dialog State Tracking Challenge (DSTC) is an on-going series of research community challenge tasks. Each task released dialog data labeled with dialog state information, such as the user’s desired restaurant search query given all of the dialog history up to the current turn. The challenge is to create a “tracker” that can predict the dialog state for new dialogs. In each challenge, trackers are evaluated using held-out dialog data.

### DSTC1
<a id="markdown-dstc1" name="dstc1"></a>

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
  
  ## A Survey of Available Corpora for Building Data-Driven Dialogue Systems
  
  - 把所有的数据集按照不同类别进行分类总结，里面涵盖了很多数据集
  - [链接](http://link.zhihu.com/?target=https%3A//docs.google.com/spreadsheets/d/1SJ4XV6NIEl_ReF1odYBRXs0q6mTkedoygY3kLMPjcP8/pubhtml)

# Resource
<a id="markdown-resource" name="resource"></a>

+ pass



# Metric
<a id="markdown-metric" name="metric"></a>



# Solutions
<a id="markdown-solutions" name="solutions"></a>

## Chat-Bot
<a id="markdown-chat-bot" name="chat-bot"></a>

## IR-Bot
<a id="markdown-ir-bot" name="ir-bot"></a>

## QA-Bot
<a id="markdown-qa-bot" name="qa-bot"></a>

## KB-Bot
<a id="markdown-kb-bot" name="kb-bot"></a>

## Task-Bot
<a id="markdown-task-bot" name="task-bot"></a>

## Pipeline
<a id="markdown-pipeline" name="pipeline"></a>

### ASR
<a id="markdown-asr" name="asr"></a>

- APIs or Tools for free

### NLU
<a id="markdown-nlu" name="nlu"></a>

- Domain CLF
  - context based domain clf
- Intent Detection
- Slot Filling
- Joint Learning and Ranking

### DM
<a id="markdown-dm" name="dm"></a>

- DST
- DPL

### NLG
<a id="markdown-nlg" name="nlg"></a>

### TTS
<a id="markdown-tts" name="tts"></a>


# Problem
<a id="markdown-problem" name="problem"></a>

# Open Issues
<a id="markdown-open-issues" name="open-issues"></a>

# Milestone
<a id="markdown-milestone" name="milestone"></a>


# Coding Standards
<a id="markdown-coding-standards" name="coding-standards"></a>

# Usages
<a id="markdown-usages" name="usages"></a>

# Reference
<a id="markdown-reference" name="reference"></a>

## Links
<a id="markdown-links" name="links"></a>

- [AIComp Top](https://github.com/linxid/AICompTop)
- [Robot Design](https://github.com/qhduan/ConversationalRobotDesign)
- [sizhi bot](https://github.com/ownthink/robot]
- [home assistant](https://github.com/home-assistant/home-assistant)
- [textClassifier](https://github.com/jiangxinyang227/textClassifier)

## Papers
<a id="markdown-papers" name="papers"></a>

### Knowledge Aware Conversation Generation with Explainable Reasoing ever Augmented Graphs
<a id="markdown-knowledge-aware-conversation-generation-with-explainable-reasoing-ever-augmented-graphs" name="knowledge-aware-conversation-generation-with-explainable-reasoing-ever-augmented-graphs"></a>
+ EMNLP 2019 Baidu
+ Tips 
  + 大部分模型 容易出现安全回复和不连贯回复，这是因为仅仅从语料中学习语义而不是借助背景知识
  + 引入结构化信息
    + 利用三元组或者图路径来缩小知识的候选范围并增强模型的泛化能力
    + 但是选出的知识往往是实体或普通词，因而无法为回复生成更加丰富的信息
  + 引入非结构化信息
    + 文本知识(电影评论或者电影剧情)可以为回复的生成提供丰富的参考信息，但是非结构化的表示方案要求模型具有很强的能力来从知识文本集合中进行知识选择或者使用注意力机制
+ 此文综合使用结构化信息和非结构化信息， 提出了基于扩充知识图(Augmented Knowledge Graph)开放域对话生成模型，模型由知识选择和回复生成这两个模块组成

## Projects
<a id="markdown-projects" name="projects"></a>

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
<a id="markdown-personalizing-dialogue-agents-i-have-a-dog-do-you-have-pets-too" name="personalizing-dialogue-agents-i-have-a-dog-do-you-have-pets-too"></a>

- 本文是 Facebook AI Research 发表于 NIPS 2018 的工作。论文根据一个名为 PERSONA-CHAT 的对话数据集来训练基于 Profile 的聊天机器人，该数据集包含超过 16 万条对话。
- 本文致力于解决以下问题：
- 聊天机器人缺乏一致性格特征
  - 聊天机器人缺乏长期记忆
  - 聊天机器人经常给出模糊的回应，例如 I don't know
- 数据集链接
- https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat

### Seq2seq code build
<a id="markdown-seq2seq-code-build" name="seq2seq-code-build"></a>
+ https://blog.csdn.net/Irving_zhang/article/details/79088143

https://github.com/qhduan/ConversationalRobotDesign/blob/master/%E5%90%84%E7%A7%8D%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%B9%B3%E5%8F%B0%E8%B0%83%E7%A0%94.md

### DeepQA
<a id="markdown-deepqa" name="deepqa"></a>
+ Seq2Seq
+ https://zhuanlan.zhihu.com/p/29075764

### RasaHQ
<a id="markdown-rasahq" name="rasahq"></a>

### NLG/GAN
<a id="markdown-nlggan" name="nlggan"></a>

### Dual Training
<a id="markdown-dual-training" name="dual-training"></a>

### More Deep
<a id="markdown-more-deep" name="more-deep"></a>
- 在可以收敛的情况下，尽可能使用更深的模型
- 参考CV 领域的一些做法
- https://zhuanlan.zhihu.com/p/35317776
- https://zhuanlan.zhihu.com/p/29967933
- https://blog.csdn.net/Irving_zhang/article/details/79088143

### A Neural Conversation Model
<a id="markdown-a-neural-conversation-model" name="a-neural-conversation-model"></a>
+ https://arxiv.org/abs/1506.05869

