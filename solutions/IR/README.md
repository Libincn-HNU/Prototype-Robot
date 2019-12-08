## IR-Bot
+ 主流的方法分为两类，一种是弱相关模型，包括DSSM，ARC-I等方法，另一种是强相关模型，包括ARC-II， MatchPyramid，DeepMatch等算法，两种方法最主要的区别在于对句子<X,Y> 的建模不同，前者是单独建模，后者是联合建模

### DSSM

#### 预处理
+ 英文 word hanshing
  + 以三个字母 切分英文单词，转化后为30k
+ 中文 子向量 15k 个左右常用字

#### 表示层
+ 原始的DSSM　用　BOW ，　后续的其他方法（CNN－DSSM　和　LSTM-DSSM 会有改进）
+ 多层DNN 进行信息表示

#### 匹配层

+ https://www.cnblogs.com/wmx24/p/10157154.html

![DSSM-1.PNG](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/DSSM-1.PNG)

#### 优缺点
+ 优点：DSSM 用字向量作为输入既可以减少切词的依赖，又可以提高模型的泛化能力，因为每个汉字所能表达的语义是可以复用的。另一方面，传统的输入层是用 Embedding 的方式（如 Word2Vec 的词向量）或者主题模型的方式（如 LDA 的主题向量）来直接做词的映射，再把各个词的向量累加或者拼接起来，由于 Word2Vec 和 LDA 都是无监督的训练，这样会给整个模型引入误差，DSSM 采用统一的有监督训练，不需要在中间过程做无监督模型的映射，因此精准度会比较高。
+ 缺点：上文提到 DSSM 采用词袋模型（BOW），因此丧失了语序信息和上下文信息。另一方面，DSSM 采用弱监督、端到端的模型，预测结果不可控。

### ARC-I and ARC-II
+ https://arxiv.org/pdf/1503.03244.pdf

### Match Pyramid

### SMN

### DMN

### 基于检索的闲聊系统的实现
+ 使用检索引擎（如ES）对所有预料进行粗粒度的排序
  + 使用Okapi BM２５　算法

+ 使用匹配算法对答案进行精排
