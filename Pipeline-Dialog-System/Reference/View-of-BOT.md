# Paper

+ The Dialog State Tracking Challenge Series: A Review
+ MACHINE LEARNING FOR DIALOG STATE TRACKING: A REVIEW

+ [A Survey of Available Corpora for Building Data-Driven Dialogue Systems](http://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1512.05742.pdf)
+ [A Neural Conversational Model](http://link.zhihu.com/?target=http%3A//cn.arxiv.org/pdf/1506.05869.pdf)
+ [A Diversity-Promoting Objective Function for Neural Conversation Models](http://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1510.03055v1.pdf)
+ [A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1605.06069v3.pdf)
+ [Sequence to Backward and Forward Sequences: A Content-Introducing Approach to Generative Short-Text Conversation](http://link.zhihu.com/?target=http%3A//cn.arxiv.org/pdf/1607.00970)
+ [A Persona-Based Neural Conversation Model](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1603.06155.pdf)
+ [Deep Reinforcement Learning for Dialogue Generation](http://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1606.01541v3.pdf)
+ [End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning](http://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1606.01269v1.pdf)
+ [A Network-based End-to-End Trainable Task-oriented Dialogue System](http://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1604.04562v2.pdf)
+ [Incorporating Unstructured Textual Knowledge Sources into Neural Dialogue Systems](http://link.zhihu.com/?target=http%3A//media.wix.com/ugd/b6d786_137894b7b3a341a09ed0c0b45b46dbb6.pdf)
+ [A Neural Network Approach to Context-Sensitive Generation of Conversational Responses](http://link.zhihu.com/?target=https%3A//michaelauli.github.io/papers/chitchat.pdf)
+ [A Dataset for Research on Short-Text Conversation](http://link.zhihu.com/?target=http%3A//staff.ustc.edu.cn/%7Echeneh/paper_pdf/2013/HaoWang.pdf)
+ [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1506.08909v3.pdf)



# People and Organization

+ [Cambridge Dialogue Systems Group](http://link.zhihu.com/?target=http%3A//mi.eng.cam.ac.uk/research/dialogue/)
+ [Huawei NOAH’S ARK LAB](http://link.zhihu.com/?target=http%3A//www.noahlab.com.hk/topics/ShortTextConversation)
+ [Jiwei Li](http://link.zhihu.com/?target=http%3A//web.stanford.edu/%7Ejiweil/)
+ [Ryan Lowe](http://link.zhihu.com/?target=http%3A//cs.mcgill.ca/%7Erlowe1/)



# Projects

+ https://github.com/vyraun/chatbot-MemN2N-tensorflow.git

# Reference

+ https://www.leiphone.com/news/201704/8ofqhd0UEJVjgWHv.html
  + 开放域-闲聊
  + 任务型
  + 问答
  + 推荐

- 如何成为一名对话系统工程师
  - https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/78746434
- 基于深度学习的对话系统
  - https://zhuanlan.zhihu.com/c_143965981
- 专栏文章导航&&对话系统梳理
  - https://zhuanlan.zhihu.com/p/33219577
- 深度学习对话系统理论篇--HRED+VHRED+Attention With Intention
  - https://zhuanlan.zhihu.com/p/33526045
- 多轮对话状态追踪（DST）--模型介绍篇
  - https://zhuanlan.zhihu.com/p/40988001
- 深度学习对话系统理论篇--MMI模型
  - https://zhuanlan.zhihu.com/p/33226848
  - “A Diversity-Promoting Objective Function for Neural Conversation Models”阅读笔记
- 深度学习对话系统理论篇--数据集和评价指标介绍
  - https://zhuanlan.zhihu.com/p/33088748
- 对话系统概述
  - https://zhuanlan.zhihu.com/p/31828371
  - Chit-Chat-oriented Dialogue Systems： 闲聊型对话机器人，产生有意义且丰富的响应。
  - Rule-based system：对话经过预定义的规则（关键词、if-else、机器学习方法等）处理，然后执行相应的操作，产生回复。（ELIZA系统，如果输入语句中没有发现预定义规则，则生成generic的响应）。缺点是规则的定义，系统越复杂规则也越多，而且其无法理解人类语言，也无法生成有意义的自然语言对话。处在比较浅层的阶段；
  - IR-based Systems：信息检索或者最近邻方法，要求生成的响应与对话存在语义相关性（VSM、TF-IDF、page-Rank、推荐等排序方法）。有点是比生成模型简单，直接从训练集中选择答案，且可以添加自定义规则干预排序函数较为灵活；缺点是无法应对自然语言的多变性、语境解构、连贯性等，对语义的细微差别也无法识别；
  - Generation-based Systems：将对话视为input-output mapping问题，提出了MT-based方法（SMT统计机器翻译、IBM-model、phrase-based MT等），这种方法复杂且无法很好的解决输入输出的对应关系（尤其是当句子较复杂的时候，只适合单词级别）。但是最近的NN、seq-to-seq等方法很好地解决了这些问题，可以生成更加丰富、有意义、特别的对话响应。但是还存在许多问题，比如沉闷的回应、agent没有一个固定的风格、多轮对话等等
  - Frame-based Dialogue Systems：定义一个对话的框架，及其中所涉及的重要元素。优点是目标明确框架对对话指导意义明显，适用于飞机票、餐馆等预定领域。缺点是框架设计需要人工成本，且无法迁移到别的领域，并未涉及到人类语言的理解层面。
  - Finite-State Machine Systems有限状态机：（用户使用预定义的模板提问，系统之响应能力范围之内的问题），这种方法的缺点是完全依赖于对框架slot的填充，而无法决定对话的进程和状态（用户接受建议、拒绝等）
  - State-based Systems：主要包含系统状态（上下文信息、用户意图、对话进程等）和系统行动两（基于state采取action）个部分。MDP、POMDP等模型。
  - Question-Answering (QA) Based Dialogue Systems：factoid QA-based，个人助手，需要回答各种各样的问题并且进行交互式对话。目前的研究点主要包括，bot如何通过对话进行自学习、对于out-of-vocab的词汇应该学会问，即学会与人交流、如何通过在线反馈学习（犯错时调整、正确时加强）
- 对话系统下的口语语义理解
  - https://speechlab.sjtu.edu.cn/pages/sz128/homepage/year/08/21/SLU-review-introduction/
- 对话系统原理和实践
  - https://blog.csdn.net/m0epnwstyk4/article/details/79285961
- 对话管理的一些思考
  - https://yq.aliyun.com/articles/276269
- 知乎专栏:基于深度学习的对话系统
  - https://zhuanlan.zhihu.com/c_143965981
  - https://blog.csdn.net/irving_zhang/article/details/78865708
- 李林琳&赵世奇.对话系统任务综述与基于POMDP的对话系统. PaperWeekly
- 多轮对话之对话管理(Dialog Management)-徐阿衡
  - https://zhuanlan.zhihu.com/p/32716205
- 浅谈垂直领域的chatbot
  - http://www.crownpku.com/2017/09/27/%E6%B5%85%E8%B0%88%E5%9E%82%E7%9B%B4%E9%A2%86%E5%9F%9F%E7%9A%84chatbot.html
- http://link.zhihu.com/?target=http%3A//lwc.daanvanesch.nl/openaccess.php)
- 使聊天机器人的对话更有营养
  - https://blog.csdn.net/aliceyangxi1987/article/details/76128058

### Others

- [A Survey of Available Corpora for Building Data-Driven Dialogue Systems](http://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1512.05742.pdf)

- 深度学习对话系统理论篇--数据集和评价指标介绍
  - https://zhuanlan.zhihu.com/p/33088748
- 如何构造数据, 标注数据
  - https://zhuanlan.zhihu.com/p/30689725
    - 作者为每句话标注了intention和emotion。在intention上作者遵循了 Amanova et al. (2016) 的方法，它是基于国际主流标准ISO 24617-2制定的
    - 这篇论文将intention分为{Inform, Questions,Directives, Commissive}四个方面
    - Inform指通知类意图，Questions指疑问类意图，Directives指建议类的意图，Commissive指接受，拒绝之类的意图。在emotion上作者遵循(Wang et al., 2013)的方法，在该方法中将情绪分为了7类: {Anger, Disgust, Fear, Happiness, Sadness, Surprise}，作者在此基础上增加了other类
- 英文纠错程序
  - https://github.com/phatpiglet/autocorrect/
- Domain-Specific Datasets：
  - TRAINS (Ringger et al., 1996) ：主要包含在购物领域的问题和解决方法的对话，通常用语训练task-oranted对话系统。
  - bAbI synthetic dialog (Bordes and Weston, 2016) and Movie Dialog datasets (Dodge
    et al., 2015)：主要是餐厅点餐和电影订票方面的对话数据
  - Ubuntu dataset (Lowe et al., 2015)：这个数据库主要是由ubuntu系统的维护者和使用者在网站上的疑难问题的对话日志组成
- Open-Domain Datasets：（闲聊）
  - Sina Weibo dataset (Wang et al., 2013)&Twitter dataset：爬取大量的微博和下面的回复作为对话
  - OpenSubtitle (Jorg Tiedemann ¨ , 2009) &SubTle dataset (Bordes and Weston,
    2016)：由大量电影的字幕构造的对话集（每3句做一个分割，第3句作为回复）
- Others
  - DailyDialog
    - http://yanran.li/dailydialog
    - https://github.com/Sanghoon94/DailyDialogue-Parser
    - http://yanran.li/files/ijcnlp_dailydialog.zip
  - A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset
    - https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/
  - https://research.fb.com/downloads/babi/

# 冷启动

- 利用规则
- 只有词槽
- 没有用户画像，没有接口槽

# Metric

- [检索回复](https://zhuanlan.zhihu.com/p/30689725)
  - Embedding-based Similarity for Response Retrieval
  - Feature-based Similarity for Response Retrieval
  - Feature-based Similarity for Response Retrieval and Reranking
  - Neural network-based for Response Generation
  - Neural network-based for Response Generation with Labeling Information
- BLEU
- Reranking
- 深度学习对话系统理论篇--数据集和评价指标介绍
  - https://zhuanlan.zhihu.com/p/33088748

# Projects

- 系统demo 的数据和代码路径：
   <https://github.com/zqhZY/_rasa_chatbot>
- 项目依赖安装（包括rasa nlu 和 rasa core）参考相应路径：
   <https://github.com/zqhZY/_rasa_chatbot/blob/master/INSTALL.md>
- 关于rasa nlu的使用方法，可以参考：
   <https://github.com/RasaHQ/rasa_nlu>
   <http://www.crownpku.com/2017/07/27/>用Rasa_NLU构建自己的中文NLU系统.html
- Microsoft : End2End Task Completion Neural Task System
  - DM : DST+DPO->RL
  - https:/github.com/MiuLab/Tc-Bot
- DST
  - https://github.com/voicy-ai/DialogStateTracking
    - https://research.fb.com/downloads/babi/
  - https://github.com/vyraun/chatbot-MemN2N-tensorflow
  - https://github.com/CallumMain/DNN-DST
  - DSTC6
    - https://github.com/perezjln/dstc6-goal-oriented-end-to-end
    - End-to-End Goal Oriented Dialog Learning
    - End-to-End Conversation Modeling
    - Dialogue Breakdown Detection
  - DSTC5
    - https://github.com/seokhwankim/dstc5
  - https://github.com/TakuyaHiraoka/Dialogue-State-Tracking-using-LSTM
- OpenDial

# Papers

- SPEAKER-SENSITIVE DUAL MEMORY NETWORKS FOR MULTI-TURN SLOT
  TAGGING

  - End2End

- Review of spoken dialogue systems

- POMDP-based statistical spokendialog systems: A review

- A Survey on Dialog System:Recent Advances and New Frontiers

  - https://arxiv.org/pdf/1711.01731.pdf

- SMN

  - 检索式多轮闲聊

- DAM

  - 检索式多轮对话
  - Attention is all your need

- A User Simulator for Task-Completion Dialogues

- ### Feudal Reinforcement Learning for Dialogue Management in Large Domains

  - 本文来自剑桥大学和 PolyAI，论文提出了一种新的强化学习方法来解决对话策略的优化问题
  - https://www.paperweekly.site/papers/1756

  ### End-to-End Optimization of Task-Oriented Dialogue Model with Deep Reinforcement Learning

  - 一篇基于强化学习的端到端对话系统研究工作，来自 CMU 和 Google。
  - 论文链接：http://www.paperweekly.site/papers/1257

  ### Learning to Ask Question in Open-domain Conversational Systems with Typed Decoders

  - ACL 2018 黄民烈
  - 深度对话模型问题:语义理解问题, 上下文理解问题, 个性身份一致性问题
  - 通过向用户提问, 能够将对话更好的进行下去
  - 提出一个好问题, 也体现了机器对人类语言的理解能力
  - 一个问题包括: interrogative(询问词), topic word(主题词), ordinary word(普通词)
  - 基于　encoder-decoder 的框架, 提出两种decoders(STD和HTD), 来估计生成出的句子中每个位置这三种词的分布

  - SIGIR 2018 | 通过深度模型加深和拓宽聊天话题，让你与机器多聊两句
    - 目前大多数基于生成的对话系统都会有很多回答让人觉得呆板无趣，无法进行有意思的长时间聊天。近日，山东大学和清华大学的研究者联合提出了一种使用深度模型来对话题进行延展和深入的方法 DAWnet。
    - 该方法能有效地让多轮对话系统给出的答复更加生动有趣，从而有助于实现人与机器的长时间聊天对话。机器之心对该研究论文进行了摘要编译。此外，研究者还公布了他们在本论文中所构建的数据集以及相关代码和参数设置
    - 论文、数据和代码地址：https://sigirdawnet.wixsite.com/dawnet

# Others

- 提供语义表达的期望值（expectations for interpretation)
- 实际实现中，对话管理模块因为肩负着大量杂活的任务，是跟使用需求强绑定的，大部分使用规则系统，实现和维护都比较繁琐。
- **比较新的研究中，有将对话管理的状态建模成为一个序列标注的监督学习问题，甚至有用强化学习(Reinforcement Learning)，加入一个User Simulator来将对话管理训练成一个深度学习的模型**
  - 首先，增强学习器尤其需要和环境进行交互，所以对话型的语料不能被直接使用。第二，每个任务代表一个特别的挑战，所以需要单独的语料库上面的特定注释的数据。第三，收集的人机对话和人人对话是非常昂贵的领域知识。因为构建一个大概的数据库可能耗时且昂贵，一个普遍的做法是构建一个用户模拟器基于一个示例对话。然后，可以在在线方式中训练增强学习agent随着它们与模拟器交互。对话agent在这些模拟语料上面训练好之后可以当做一个有效的起始agent。一旦agent掌握了如何与模拟器交互，他们可以被部署在真实环境中和人类进行交互，并且被继续在线训练。为了简化对话中的经验算法比较，本文引入了一种新的、公开的仿真框架，其中我们为电影预订领域设计的模拟器利用规则和收集的数据。这个模拟器支持两项人物：电影票订购和电影查询。最后，我们演示几个agent和一些如何添加并测试你自己的agent使用提出的框架

- Question answering over Freebase (single-relation)
  - https://github.com/quyingqi/kbqa-ar-smcnn

- (***)基于CNN的阅读理解式问答模型：DGCNN
  - Dilate Gated Convolutional Neural Network
  - Ref : 一文读懂「Attention is All You Need」| 附代码实现
  - 本模型——我称之为 DGCNN——是基于 CNN 和简单的 Attention 的模型，由于没有用到 RNN 结构，因此速度相当快，而且是专门为这种 WebQA 式的任务定制的，因此也相当轻量级。
  - SQUAD 排行榜前面的模型，如 AoA、R-Net 等，都用到了 RNN，并且还伴有比较复杂的注意力交互机制，而这些东西在 DGCNN 中基本都没有出现。
  - 这是一个在 GTX1060 上都可以几个小时训练完成的模型！
  - CIPS-SOGOU/WebQA

+ WebQA
  + WebQA 的参考论文 Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question :
    + 直接将问题用 LSTM 编码后得到“问题编码”，然后拼接到材料的每一个词向量中；
    + 人工提取了 2 个共现特征；
    + 将最后的预测转化为了一个序列标注任务，用 CRF 解决。                                    