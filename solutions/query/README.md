# Query Process

## 1. 问句改写

### 1.1 纠错
+ pycorrector
+ 相关总结见
    + https://blog.csdn.net/yiminghd2861/article/details/84181349

+ 基于transformer 的通用纠错

### 1.2 指代消解
+ standford core nlp
+ (PAI)Mention Pair models：将所有的指代词（短语）与所有被指代的词（短语）视作一系列pair，对每个pair二分类决策成立与否。
+ (PAI)Mention ranking models：显式地将mention作为query，对所有candidate做rank，得分最高的就被认为是指代消解项。
+ (PAI)Entity-Mention models：一种更优雅的模型，找出所有的entity及其对话上下文。根据对话上下文聚类，在同一个类中的mention消解为同一个entity。但这种方法其实用得不多。

+ 分类
    + https://blog.csdn.net/u013378306/article/details/64441596
+ 零指代消解
    + https://www.jiqizhixin.com/articles/2018-07-28-8
+ https://zhuanlan.zhihu.com/p/53550123
+ https://zhuanlan.zhihu.com/p/103794289


### 1.3 专有名替换/缩略词扩充
+ 针对具体应用场景对专有名词和缩略词进行处理
+ https://github.com/zhangyics/Chinese-abbreviation-dataset

### 1.4 省略消解

### 1.5 文本归一化
+ 时间/单位/数

### 1.7 繁简转化 
+ snow nlp

### 1.8 stopwords 去除
+ http://www.baiduguide.com/baidu-stopwords/
+ https://github.com/ourren/stopwords_zh
+ https://github.com/stopwords-iso/stopwords-zh

## 2. 长难句压缩/消除冗余/文本摘要
+ 语法树分析 加 关键词典
+ 抽取方式
+ 生成方式
+ ByteCup 基于transformer

## 3. 分词
+ jieba/ltp

## 4. 词性标注
+ ltp

## 5. 命名实体识别
+ ltp

## 6. 句法分析
+ ltp

## 7. 关键词
+ FlashText

## 8. 意图识别
+ 分类模型
+ 多意图

## 9. 情感分析
+ 判断情感，明确query 的情感倾向，作为系统反馈
+ sentiment

## 10. 拼音
+ snownlp

## 11. QQ匹配
+ 找到与query相关的question
    + 字面相似
    + 语义相似

## 12. QA匹配
+ 找到query 相关的answer（具体应用有待细化）
    + 字面相似
    + 语义相似

## 13. 语义索引
+ query 的语义表示 

## 14. 知识库查询
