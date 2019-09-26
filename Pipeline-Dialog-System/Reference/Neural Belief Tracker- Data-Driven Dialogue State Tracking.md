# Neural Belief Tracker : Data-Driven Dialogue State Tracking

### ACL 2017

### Reference

+ Steve Young, Milica Gasic, Simon Keizer, Francois Mairesse, Jost Schatzmann,
  Blaise Thomson, Kai Yu. 2010. The Hidden Information Statemodel: A practical framework for
  POMDP-based spoken dialogue management. *Computer Speech and Language* 24: 150-174.

### Background

+ Separate SLU
  +  Given enough data, these models can learn which lexical features are good indicators for a given value and can capture elements of paraphrasing (Mairesse et al., 2009). 
  + sequence labeling
  + Semantic parsing
+ Joint SLU/DST
  + pass

### Problem

+ Spoken Language Understanding models that require large amounts of annotated training data
+ hand-crafted lexicons for capturing some of the linguistic variation in users’ language

### Overcome of Problem

+ building on recent advances in representation learning
  + pretrained embedding
  + learning to compose the embedding into user utterance and dialogue text

### Belief Tracker

+ pass

### Neural Belief Tracker

#### Target

+ 在对话过程中，检测 slot-value 对， 完成用户的目标  

#### Frames

![](/Users/sunhongchao/Desktop/屏幕快照 2018-12-24 下午5.26.47.png)

#### Input

+ user utterance
  + $$r$$
+ system dialogue acts  preceding user input
  + $$t$$
+ a single candidate slot-value pair that it needs to make decision
  + $$c$$
+ Demo
  + I'm looking for good pizza 
  + FOOD=pizza
  +  判断第一句话是否表示可以用第二句中 的 slot-value 对来表示

#### Represent Learing

+ NBT-DNN
+ NBT-CNN

#### Semantic Decoding

+ $$d$$
+ $$d_r$$, $$d_c$$, $$d$$
+ $$r$$ and $$c$$ directly interact through the semantic decoding module
+ 决定用户是否表达了 current candidate pair 
+ 并没有考虑dialogue context
+ Demo
  + 'I want Thai food' match 'food=Thai'
  + 'a pricey restaurent' match 'price=expensive'
+ 需要高质量的 pre-trained word vectors
  +  Delexicalisation-based model 可以处理前面的例子， 但是不能处理latter case
+ Slot : $$c_s$$ , Value:$$c_v$$
  + 多个词组成的 slot 和 value 采用 sum 的方式处理
+ map（$$c_s, c_v)$$ into sinlge vector c
  + $$ c = \delta(W_{c}^{s}(c_s + c_v) + b_{c}^{s})$$
+ match r and c
  + $$d = r \bigotimes c$$
  + $$\bigotimes$$ denotes element-wise vector multiplication

#### Context Modeling

+ Semantic Decoding 因为并未考虑context， 所以还需要Context Modeling来进一步处理

+ 为了理解一些query, belief tracker 必须考虑最近的user utterance
  + System Request(针对缺失的槽值 $$T_q$$ 进行询问)
    - system aks the user about the vualue of special slot **$$T_q$$**

    - ```
      - Bot  ：what price range would you like？
      - User ：any
      ```

    - $$T_q$$ :  提问的槽

  + System Confirm(确认对应的槽值)

    + ```
      - BOT ：How about Thai Food
      - User ：Yes
      ```

    + $$T_s $$ :  确认的槽

    + $$T_v$$ ：槽的值

- Make the Markovian decision to only consider the last set of system
  - change $$T_q, (T_s, T_v)$$  into word vector for system request and confirm acts
  - zero vectors if none
- 计算 system acts, candidate pair$$(c_s, c_v)$$, utterance 的相似性
  - $$m_r = (c_s \cdot t_q) r$$
  - $$m_c = (c_s \cdot t_s) (c_v \cdot t_v) r$$
  - $$\cdot$$  denote dot predict

#### Binary Decision Maker

+ $$\phi_{dim}(x) = \delta(Wx + b)$$
  + 将输入x 映射到 dim 维度
+ $$ y = \phi_{2}(\phi_{100}(d) + \phi_{100}(m_r) + \phi_{100}(m_c))$$

### Belief State Update Mechanism

+ 处理 ASR N-Best 最佳结果
+ Dialog turn t
  + $$sys^{t-1}$$ : system output
  + $$h$$ : n ASR hypotheses
  + 对于任何$$h^t_i$$, 对应的slot 和 value 又以下公式确定
    + NBT model estimate $$P(s,v|h^t_{i}, sys^{t-1})$$
  + 将ASR识别的所有结果组合起来，得到(s, v) 的概率值
    + $$P(s,v|h^{t}, sys^{t-1} ) =\sum^{N}_{t-1} p^{t}_{i} P(s,v|h^t_{i}, sys^{t-1}) $$ 
    + 原始论文中 最后一个 sys 的上标为 t， 不理解
  + turn-level
    + $$P(s,v|h^{1:t}, sys^{1:t-1}) =\lambda P(s,v|h^t, sys^{t-1}) + (1-\lambda) P(s,v|h^{1:t-1}, sys^{1:t-2}) $$
    + $$\lambda $$  是超参数
  + 在turn t， 对某slot s，计算出values 满足
    + $$V_s^t = \{v\in V_s |  P(s,v|h^{1:t}, sys^{1:t-1})  > 0.5 \}$$ 
  + For informable slots(包含信息的槽)
    + 最高概率值的value最为当前的goal
  + For request， all slots in  $$V^t_{rcq}$$ are deemed to have bee requested
    + 视为是单轮查询，不需要进行多轮处理 

### Experiment

+ Data
  + WOZ
  + DSTC2

