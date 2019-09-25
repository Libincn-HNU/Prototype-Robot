[TOC]

# Target

### Tricks

+ 利用多轮的上下文信息，用户画像信息（更新），澄清话术（交互）
+ 提升分类器性能
+ 提升intent和slot 提取的性能

### Context

+ 记录前两句的数据
+ 输入
  + 前两句的文本
  + 当前句的文本
+ 记录前两句的domain, intent, slot
+ 利用前两句的domain, intent, slot 提取当前句的domain, intent, slot
  + 当前句的domain（分类）
    + 当前句的话术
    + 前两句的话术
    + 前两句的domain和intent(利用现有domain 分类器 来做标注)

### User Portrait

+ Pass

### Clarification

+ Pass



![](http://img.pmcaff.com/FomDNcyHrznIM_ccW-Y5L0eO6adu-picture)

![](http://www.crownpku.com/images/201709/5.jpg)

# Dialogue Manger

+ Task 1 : Dialog State Update
  + Dialog state encode every information relevant to the system
  + DM maintain a representation of current dialogue state
  + DM update the dialogue state as new information coming
+ Task 2 : Action Selection

  + Make decision on which actions to take on dialogue state
+ Outcome of Dialogue State
  + Update dialogue state
  + One or more select system action(s)

+ ### 分类

  - 开放域
    - 目的：识别用户意图，进入封闭域
  - 封闭域
    - 输入和输出是可以枚举的
    - 对话有明确的目的，且有流程

  ### Initiative

  - 系统主导
  - 用户主导
  - 混合

  ### Challenges

  人的复杂性（complex）、随机性（random）和非理性化（illogical）的特点导致对话管理在应用场景下面临着各种各样的**问题**，包括但不仅限于：

  - **模型描述能力与模型复杂度的权衡**
  - **用户对话偏离业务设计的路径**
    如系统问用户导航目的地的时候，用户反问了一句某地天气情况
  - **多轮对话的容错性**
    如 3 轮对话的场景，用户已经完成 2 轮，第 3 轮由于ASR或者NLU错误，导致前功尽弃，这样用户体验就非常差
  - **多场景的切换和恢复**
    绝大多数业务并不是单一场景，场景的切换与恢复即能作为亮点，也能作为容错手段之一
  - **降低交互变更难度，适应业务迅速变化**
  - **跨场景信息继承**

+ DM 设计

  - 一般在冷启动阶段，会先用规则方法打造一个 DM，快速上线并满足业务需求，收集数据之后再转换成模型。

  - DM 的**设计理念：**

  - **完整性**
    + 具备建模各种类型对话的能力（不仅仅是slot filling）
  - **独立性**
    + 当设计（变更）一个场景时，不需要考虑当前场景跳转到其他场景的情况
  - **模块化**
    + 一些常用的业务组件(如：确认，slot filling，翻页等)，能呈模块化复用(同时允许业务自定义内部的多种属性)

+ DM 里可以有很多小的 dialogs，这些 dialogs 的特点是：

  - 可以重用（reusable modules）
  - 完成一个简单操作（Perform a single operation）
  - 可以被其他 dialog 调用（callable from other dialogs）
  - 可以是全局的（”global”）

+ Global dialog 的特点是：

  - 在 recognizer 能够匹配时 trigger
  - 可以提供一些 conversation support，像是 help/cancel 这些全局功能
  - 应对各种话题切换（Tangents）

+ 通常只有一个 root dialog，在满足下面两个条件的情况下被唤醒

  - dialog stack 里没有其他的 dialog 剩余了
  - 当前时刻 recognizer 并不能 trigger 其他的 dialog

+ dialog stack 会存放目前已经被激活但还没完成的 dialog。dialog 一旦完成就会被 pop 出去

# Dialogue state tracking

## Define of state

+ 状态表示
+ 状态跟踪

## Input and output

+ input
  + $$U_n$$ : n 时刻的意图和槽值对，也叫用户Action
  + $$A_{n-1}$$ : n-1 时刻的系统Action
  + $$S_{n-1}$$ : n-1 时刻的状态

+ Output

  +  $$S_n$$ : n 时刻的状态
  + $$ S_n = \{G_n, H_n, H_n\}$$ 
    + $$G_n$$ 是用户目标
    + $$U_n$$ 同上
    + $$H_n$$ 是聊天历史
      + $$H_n= U_0, A_0, U_1, A_1, ..., U_{n-1}, A_{n-1}$$
  + $$ S_n = f(S_{n-1}, A_{n-1}, U_n)$$ 


## Rule Methods

### Key Pharse Reactive Approaches

- 本质是关键词匹配
- 通常是捕获用户的最后一句话的**关键词/关键短语**来进行回应
- ELIZA，AIML（AI标记语言）
- 能支持一定的上下文实现简单的多轮对话
- https://github.com/Shuang0420/aiml
  - 支持python3
  - 支持中文
  - 支持*拓展

## Structure Methods

### Overview of Structure Methods

![](http://ata2-img.cn-hangzhou.img-pub.aliyun-inc.com/fd3925caf61526910ed7145058a05635.png)

### Trees-based: Approaches

- 通常把对话建模为通过树或者有限状态机（图结构）的路径
- 相比 **simple reactive approach**, 这种方法融合了更多的上下文，能用一组**有限的信息交换模板**来完成对话建模
- 适用于
  - **系统主导**
  - **需要从用户搜集特定的信息**
  - 用户对每个问题的答案在**有限的集合中**

### Graph-Base ：FSM（Finite State Machine)

- 把对话在有限的状态内跳转的过程，**每个状态都有对应的动作和回复**

- 如果能从开始节点顺利流转到终止节点，那就认为任务完成

  ![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/fsa.png)

  ![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/credit_card_eg.png)

- 特点：

  - 人为定义对话流程
  - **完全由系统主导，系统问，用户答**
  - 答非所问的情况直接忽略
  - 建模简单，能清晰明了的把交互匹配到模型
  - **难以扩展，很容易变得复杂**
  - 适用于简单任务，对简单信息获取很友好，难以处理复杂的问题
  - 缺少灵活性，表达能力有限，输入受限，对话结构/流转路径受限
  - 对特定领域要设计 task-specific FSM，简单的任务 FSM 可以比较轻松的搞定，但稍复杂的问题就困难了，毕竟要考虑对话中的各种可能组合，编写和维护都要细节导向，非常耗时。一旦要扩展 FSM，哪怕只是去 handle 一个新的 observation，都要考虑很多问题。实际中，通常会加入其它机制（如变量等）来扩展 FSM 的表达能力

  ![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaYrcCKVibnyhDlwqYY1b3T5UDwLf77wh8hH8XW6rNxoNjNqZIwaicl3CnXChz1ko0yGT9nDUeh5octsVlDnpydoQ/640?wx_fmt=png)

## Principle-based Methods

### Frame-based Approaches

- **通过允许多条路径更灵活的获取信息，拓展了FSM**
- 将对话构建成一个**填槽**的过程
- 槽就是多轮对话过程中将初步用户意图转化为明确用户指令所需要补全的信息
- 一个槽与任务处理中所需要获取的一种信息相对应
- 槽直接没有顺序，缺什么槽就向用户询问对应的信息

![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/slot_filling.png)

- 要素
  - Frame：槽位的集合，定义了需要用户提供什么信息
  - 对话状态：记录了那些槽位已经被填充了
  - 行为选择：下一步该做什么，填充什么槽位，还是进行何种操作
    - 行为选择可以按照槽位填充/槽位加权填充，或者利用本体选择
- 基于**框架/模板**的系统本质上是一个生成系统
  - 不同类型的输入激发不同的生成规则
  - 每个生成能够灵活填入对应的模板
  - 常常用于用户可能采取的行为相对有限，只希望用户在这些行为进行少许转化的场合
- 特点
  - 用户回答可以包含任何一个片段/全部的槽信息
  - 系统来决定下一个行为
  - 支持混合主导型系统
  - 相对灵活的输入，支持多种输入/多种顺序
  - 适用于相对复杂的信息获取
  - 难以应对更复杂的情境
  - 缺少层次
- 更多可参考[槽位填充与多轮对话](https://coffee.pmcaff.com/article/971158746030208/pmcaff?utm_source=forum&from=related&pmc_param%5Bentry_id%5D=950709304427648)

### Agenda + Frame(CMU Communicator)

- 对frame model 进行了改进
- 有了层次结构
- 能应对更复杂的信息获取
- **能支持话题切换、回退、退出**
- 要素
  - product:反应了System 与 BOT 都同意的信息
    - 树的结构
    - 反应为完成这个任务需要所有信息的顺序
    - 相比Tree and FSM approach, 这里的产品数(product tree)的创新在于它是动态的，可以在session中对树进行一系列操作比如加一个子树或挪动子树
  - agenda:确定与主题相关的任务并去完成
  - handler:在紧密耦合的信息集上管理对话行为
- AN AGENDA-BASED DIALOG MANAGEMENT ARCHITECTURE FOR SPOKEN LANGUAGE SYSTEMS
  - http://link.zhihu.com/?target=http%3A//www.cs.cmu.edu/%7Exw/asru99-agenda.pdf
- CMU Communicator
  - http://www.speech.cs.cmu.edu/Communicator/index.html
- https://blog.csdn.net/weixin_38358881/article/details/81868346

### RavenClaw

+ <http://www.cs.brandeis.edu/~cs115/CS115_docs/RavenClaw_Hierarchical_tasks.pdf>

### Information State Approaches

+ Information State-Based Dialogue Management

+ An ISU Dialogue System Exhibiting Reinforcement Learning of Dialogue Policies:Generic
  + This prototype is the first “InformationState Update”(ISU)dialoguesystemtoexhibitreinforcement

- 提出的背景

  - 很难去评估各种DM系统
  - 理论和实践模型存在很大的gap
    - 理论的模型
      - logic-based, BDI, plan-based, attention/intention
    - 实践的模型
      - finite-state 或者 frame-based

- Information State Models 作为对话建模的形式化理论，为工程化实现提供了理论指导

- 为改进当前的对话系统提供了大的方向

- Information-state theory 的关键是识别对话中流转信息的relevant aspects

  - 这些成分是如何被更新
  - 更新过程如何控制

- 理论框架

  ![](https://pic2.zhimg.com/80/v2-bd700b2e509e7d2d84a8ffad91a9ce55_hd.jpg)

- 要素

  - Statics
    - Informational components
    - Formal representaions
  - Dynamics
    - dialog move
    - update rules
    - update strategy

- 意义：

  - 可以遵循这一理论体系来构建/分析/评价/改进对话系统
- 基于information state的系统有
  - TrindKit Systems
    - GoDis
    - MIDAS
    - EDIS
    - SRI Autoroute
  - Successor Toolkits
    - Dipper
    - Midiki
  - Other IS approaches
    - Soar
    - AT&T MATCH system

- Dialogue Move Engine

### Plan-based Approaches

- BDI(Belief, Desire, Intention)模型
  - Cohen and Perrault 1979
  - Perrault and Allen 1980
  - Allen and Perrault 1980
- 基本假设
  - Cohen and Perrault 1979 提到的 **AI Plan model** : 一个试图发现信息的行为人，能够利用标准的plan找到让听话人找到说话人该信息的plan
  - Perrault and Allen 1980 和 Allen and Perrault 1980 将 BDI 应用于理解，特别是间接言语语效的理解，本质上是对 Searle 1975 的 speech acts 给出了可计算的形式体系
- 概念
  - goals
  - actions7
  - plan construction
  - plan inference

+ Back to the Future for Dialogue Research: A Position Paper
  + https://arxiv.org/pdf/1812.01144.pdf

```
However, except for hand-built examples, current virtual
assistant systems are not typically providing such assistance. In order to build systems that can engage humans in
collaborative plan-based dialogue, research is needed on
planning, plan recognition, and reasoning about people’s
mental and social states (beliefs, desires, goals, intentions,
permissions, obligations, etc.), and their relation to conventional behavior. Plan recognition involves observing actions and inferring the (structure of) reasons why those actions were performed, often to enable the actor to perform
still other actions (Allen & Perrault, 1980; Geib & Goldman, 2009; Sukthankar, et al., 2014). Belief-desire-intention (BDI) theory (Cohen & Levesque, 1990) and architectures (Bratman et al., 1988) within the subfield of MultiAgent Systems are intimately related to dialogue processing. Prior research, including our own, has developed
prototypes of the above collaborative processing, and has
shown that such collaborative BDI architectures and epistemic reasoning can form the basis for dialogue managers
(Allen & Perrault, 1980; Cohen & Perrault, 1979; Sadek et
al., 1997).
However, though the BDI approach has been researched
for many years, with few exceptions (e.g., Allen’s group at
the University of Rochester/IHMC (Galescu et al., 2018)),
it has not seen recent system application to collaborative
dialogue. Our recent plan-based dialogue manager prototype (which I will demonstrate at the workshop) uses the
same planner to reason about physical and speech acts, enabling the system to plan yes/no and wh-questions when
the user is believed to know the answers1
, to make requests
when the system wants the effect and the user is believed
to be able to perform the requested action, to inform the
user of facts the user is not believed to already know, to
suggest actions that the user may want that would further
the user’s goals, etc. 
```



## Statistical Methods

+ 基于统计的方法（A Survey on Dialogue Systems:
  Recent Advances and New Frontiers）
  - 会对每轮对话都计算每个slot的概率分布
  - robust sets of hand-crafted rules[93]
  - conditional random fields[26;25;63]
  - maximun entropy models[98]
  - web-style-ranking[101] 

### POMDP

+ 简单来说，它将对话表示成一个部分可见的马尔可夫决策过程
+ 所谓部分可见，是因为DM的输入是存在不确定性的，例如NLU的结果可能是错误的
+ 因此，对话状态不再是特定的马尔可夫链中特定的状态，而是针对所有状态的概率分布
+ 在每个状态下，系统执行某个动作都会有对应的回报（reward）
+ 基于此，在每个对话状态下，选择下一步动作的策略即为选择期望回报最大的那个动作
+ 这个方法有以下几个优点：
  + 只需定义马尔可夫决策过程中的状态和动作，状态间的转移关系可以通过学习得到；
  + 使用强化学习可以在线学习出最优的动作选择策略

+ 缺点
  + 即仍然需要人工定义状态，因此在不同的领域下该方法的通用性不强

![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaYrcCKVibnyhDlwqYY1b3T5UDwLf77wh8YLZbrB8L8lciaEjHwjDUCtEBDODCKnXoicPYFyH4C4XZicialbbVvFpPnQ/640?wx_fmt=png)

- http://www.sohu.com/a/146206452_500659

![](http://img.mp.itc.cn/upload/20170605/ca386549107c4478bda222f030dd454f_th.jpg)

## RL-Based Methods

- 前面提到的很多方法
  - 需要人工来定规则的（hand-crafted approaches），然而人很难预测所有可能的场景，这种方法也并不能重用，换个任务就需要从头再来。
  - 而一般的基于统计的方法又需要大量的数据。
  - 再者，对话系统的评估也需要花费很大的代价。
  - 这种情况下，强化学习的优势就凸显出来了。RL-Based DM 能够对系统理解用户输入的不确定性进行建模，让算法来自己学习最好的行为序列。首先利用 simulated user 模拟真实用户产生各种各样的行为（捕捉了真实用户行为的丰富性），然后由系统和 simulated user 进行交互，根据 reward function 奖励好的行为，惩罚坏的行为，优化行为序列。由于 simulated user 只用在少量的人机互动语料中训练，并没有大量数据的需求，不过 user simulation 也是个很难的任务就是了
- ![](http://images.shuang0420.com/images/NLP%E7%AC%94%E8%AE%B0%20-%20%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E4%B9%8B%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%28Dialog%20Management%29/rl_arch.png)

## DL-Based Methods

+ 基本思路
  + 它的基本思路是直接使用神经网络去学习动作选择的策略，即将NLU的输出等其他特征都作为神经网络的输入，将动作选择作为神经网络的输出。这样做的好处是，对话状态直接被神经网络的隐向量所表征，不再需要人工去显式的定义对话状态。当然这种方法的问题时需要大量的数据去训练神经网络，其实际的效果也还有待大规模应用的验证。助理来也的对话系统中有尝试用该方法，但更多的还是传统机器学习方法和基于深度学习的方法结合

![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaYrcCKVibnyhDlwqYY1b3T5UDwLf77wh8SnuXee0JpSC8FJNenrKD5P5VUfhYrmjOTEBZtokDsbnghPclibbJKWw/640?wx_fmt=png)

- 下文介绍了深度学习在信念跟踪上的应用, 在一个domain上可迁移到新的domain上
  - Deep neural network approach for the dialog state tracking challenge
- 下文提出了multi-domain RNN dialog state tracking models
  - Multidomain dialog state tracking using recurrent neural networks. 
- 下文提出了neural belief tracker(NBT)来识别slot-value对
  - Neural belief tracker: Data-driven dialogue state tracking.
- ALC 2018 **Fully NBT**
  - ![](https://pic1.zhimg.com/80/v2-f093af74079578025955f63b6bfe1c24_hd.jpg)



## Transfor Learning in DST

+ Pass

# Dialog policy optimization

## Demo

+ 线上购物的场景中，若上一步识别出的对话状态是“Recommendation”，那下一就会有推荐的action

## Papers

+ 有监督学习
  + rule-based agent 来做热启动，然后监督学习会在rule生成的action上进行
  + Learning to respond with deep neural networks for retrieval-based human computer conversation system. 

+ 端到端强化学习

  + Strategic dialogue management via deep reinforcement learning. arxiv.org, 2015

+ **End-to-End Reinforcement Learning of Dialogue Agents for Information Access**

  ```
  6.1 Policy-Value Based
  
  6.1.1 Grid based Q-function
  
  k-nearest neighbor monte-carlo control algorithm for pomdp-based dialogue systems. In Proceedings of the SIGDIAL 2009 Conference. Lefevre et al., SIGDIAL 2009
  
  
  
  6.1.2 Linear model Q-function
  
  Reinforcement learning for dialog management using least-squares policy iteration and fast feature selection. Li et al., Interspeech 2009
  
  
  
  6.1.3 Gaussian Process based Q-function
  
  Gaussian processes for pomdp-based dialogue manager optimization. IEEE/ACM Transactions on Audio, Speech, 2014. Gaši ́c et al. 2014
  
  
  
  6.1.4 Neural Network based Q-function
  
  Off-policy learning in large-scale pomdpbased dialogue systems. In 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). Daubigney et al., 2012
  
  
  
  6.2 Policy-Policy Based
  
  6.2.1 Softmax policy function
  
  Natural actor and belief critic: Reinforcement algorithm for learning parameters of dialogue systems modelled as pomdps. ACM Transactions on Speech and Language Processing (TSLP), 7(3):6, 2011. Jurcícek et al. 2011
  
  6. 2.2 Neural network policy function
  
  Continuously learning neural dialogue management. Su et al. 2016
  
  A network-based end-to-end trainable task-oriented dialogue system. Wen et al. 2016b
  
  
  
  6.3 Policy-Actor Critic
  
  6.3.1 A Q-function is used as critic and a policy function is used as actor.
  
  Sample-efficient Actor-Critic Reinforcement Learning with Supervised Data for Dialogue Management ,Su et al., SIGDIAL 2017
  
  
  
  6.4 Transfer learning for Policy
  
  6.4.1 Linear Model transfer for Q-learning
  
  Transfer learning for user adaptation in spoken dialogue systems. In Proceedings of the 2016 International Conference on Autonomous Agents & Multiagent Systems. Genevay and Laroche 2016
  
  
  
  6.4.2 Gaussian Process transfer for Q-learning
  
  Incremental on-line adaptation of POMDP-based dialogue managers to extended domains. In Proceedings of the 15th Annual Conference of the International Speech Communication 2014.
  
  POMDP-based dialogue manager adaptation to extended domains. In Proceedings of the 14th Annual Meeting of the Special Interest Group on Discourse and Dialogue, 2013.
  
  Distributed dialogue policies for multi domain statistical dialogue management. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). Gaši ́c et al. 2015
  
  
  
  6.4.3 Bayesian Committee Machine transfer for Q-learning
  
  Policy committee for adaptation in multi-domain spoken dialogue systems. 2015. Gaši ́c et al. 2015b
  
  
  
  6. 5 Neural Dialogue Manager
  
  Deep Q-network for training DM policy
  
  End-to-End Task-Completion Neural Dialogue Systems, Li et al., 2017 IJCNLP
  
  
  
  6.6 SL + RL for Sample Efficiency
  
  Sample-efficient Actor-Critic Reinforcement Learning with Supervised Data for Dialogue Management ,Su et al., SIGDIAL 2017
  
  
  
  6.7 Online Training
  
  Policy learning from real users
  
  6.7.1 Infer reward directly from dialogues
  
  Learning from Real Users: Rating Dialogue Success with Neural Networks for Reinforcement Learning in Spoken Dialogue Systems, Su et al., Interspeech 2015
  
  
  
  6. 7.2 User rating
  
  On-line Active Reward Learning for Policy Optimisation in Spoken Dialogue Systems, Su et al., ACL 2016
  
  
  
  6. 8 Interactive RL for DM
  
  Interpreting Interactive Feedback
  
  Interactive reinforcement learning for task-oriented dialogue management, Shah et al., 2016
  ```


### Frams

+  如何定义 state（what the user want）
+ 如何定义 action （what the system action）

![](https://pic1.zhimg.com/80/v2-1a9acb0290b6bef15302acab1ace9594_hd.jpg)

![](https://pic3.zhimg.com/80/v2-86717d0836eccdd4927732fcd2530302_hd.jpg)

## Define of Act

+ 用户的act
  + 对应SLU的 Domain， Intent， Slot 处理
+ 系统的act
  + 表明在限制条件下（之前积累的目标，对话历史等）系统要做的工作
  + 不追求当前收益的最大化，而是未来收益最大化

## Input and output

+ Input
  + $$ S_n = \{G_n, H_n, H_n\}$$ 
+ Output
  + $$A_n = \{ D_n, \{A_i, V_i\}\}$$
    + $$D_n$$ : 对话类型
    + $$A_i $$  和 $$ V_i$$ 是第i轮对话的attribute 和 value

 

## Demo of DST and DPL

![](https://pic4.zhimg.com/80/v2-bd6aec9b2f142166660827d5aaab3617_hd.jpg)

![](https://pic4.zhimg.com/80/v2-77cdd10bed9d338acc160fdec9400c13_hd.jpg)

![](https://pic3.zhimg.com/80/v2-acd3355d080fc290a76ce12263eef3da_hd.jpg)



## Frams

![](https://pic4.zhimg.com/80/v2-67539902b4d8f61b8986c3f3f1d18c0b_hd.jpg)

![](https://pic4.zhimg.com/80/v2-5bc09e56e620003e6d338731fae7f0f3_hd.jpg)



## Rule-based Method

## DRL-Based Method

![](https://pic4.zhimg.com/80/v2-2153a653bbf90431fce9bd006798d8db_hd.jpg)

### Value Based DPL

+ k-Nearest Neighbor Monte-Carlo Control Algorithm for POMDP-based Dialogue Systems

  + K 近邻 + 蒙特卡洛 + POMDP(部分可观察马尔可夫决策过程)
  + SIGDAL 2009
  + 适于在一些场景下的工程实现，因为本方法适合加规则和trick

  ![](https://pic1.zhimg.com/80/v2-80a0d99107e96c6cba15a3ab37f93b74_hd.jpg)

+ Gaussian processes for POMDP-based dialogue manager optimisation
  + 高斯过程  + POMDP
+ Reinforcement Learning for Dialog Management using Least-Squares Policy
  Iteration and Fast Feature Selection
  + LSPI-FFS：最小二乘+ 快速特征选择
  + 效率高，可以从静态预料库或在线学习
  + 可以更好的处理在对话模拟中评估的大量特征(???)
  +  当时的SOTA（2009）
+ OFF-POLICY LEARNING IN LARGE-SCALE POMDP-BASED DIALOGUE SYSTEMS
  + POMDP + 离线学习
  + 提出了一个样本有效，在线和非策略的方法来学习最优策略
  + 结合一个紧凑的非线性函数表示（多层感知机），能够处理大规模系统。（在线学习的方式一般规模都比较受限）

### Policy-Based DPL

+ Natural Actor and Belief Critic: Reinforcement algorithm for learning parameters of dialogue systems modelled as POMDPs
+ A Network-based End-to-End Trainable Task-oriented Dialogue System
  + 2016
  + 端到端任务型对话开创性工作，质量很高
+ Continuously Learning Neural Dialogue Management
  + 与上文是同一波作者
  + DM中的持续学习
  + 从监督学习开始，利用强化学习不断改进

### Actor Critic DPL

+ Pass

###  Transfor-based DPL

+ Pass

### Online-Leading DPL

+ Pass

### DPL Framework comparison

![](https://pic4.zhimg.com/80/v2-913f6f2a861e2b584598a582e947e5d7_hd.jpg)



## Evaluation for Dialogue Policy Learing

![](https://pic2.zhimg.com/80/v2-e7f4915993cfe36f980c4a2cff098039_hd.jpg)