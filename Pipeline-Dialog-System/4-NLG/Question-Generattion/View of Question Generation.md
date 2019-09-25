# View of Question Generation
### Some
+ QG 的应用还是挺广泛的，像是为 QA 任务产生训练数据、自动合成 FAQ 文档、自动辅导系统（automatic tutoring systems）等。 
+ 传统工作主要是利用**句法树或者知识库**，基于规则来产生问题。如基于语法（Heilman and Smith, 2010; Ali et al., 2010; Kumar et al., 2015），基于语义（Mannem et al., 2010; Lindberg et al., 2013），大多是利用规则操作句法树来形成问句。还有是基于模板（templates），定好 slot，然后从文档中找到实体来填充模板（Lindberg et al., 2013; Chali and Golestanirad, 2016）。 
### 深度学习方面的工作不多，有意思的有下面几篇： 
1. Generating factoid questions with recurrent neural networks: The 30m factoid question-answer corpus 
	+ 将 KB 三元组转化为问句 
2. Generating natural questions about an image 
	+ 从图片生成问题 
3. Semi-supervised QA with generative domain-adaptive nets 
	+ 用 domain-adaptive networks 的方法做 QA 的数据增强

### (Question Generation) Machine Comprehension by Text-to-Text Neural Question Generation
- 论文 | Machine Comprehension by Text-to-Text Neural Question Generation
- 链接 | https://www.paperweekly.site/papers/330
- 作者 | Xingdi Yuan / Tong Wang / Caglar Gulcehre / Alessandro Sordoni / Philip Bachman / Sandeep Subramanian / Saizheng Zhang / Adam Trischler