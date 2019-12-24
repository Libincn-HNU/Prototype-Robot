# Math - Matrix Factorization

### Reference
+ LHY ML 2017 Video 24


### Matrix Factorization for Topic Model
![](/home/apollo/Pictures/MF1.png)

+ 个别数据缺失
	+ 求解$r^i,r^j$，使用Gradient Descent训练
![](/home/apollo/Pictures/MF2.png)
	+ r 由一系列属性构成
	+ 由r计算缺失值
![](/home/apollo/Pictures/MF3.png)

+ 加一些偏置项
+ ![](/home/apollo/Pictures/MF4.png)


+ Latent semantic analysis(LSA)
	+ ![](/home/apollo/Pictures/TopM1.png)
+ Probability latent semantic analysis(PLSA)

+ latent Dirichlet allocation(LDA)
	+ https://blog.csdn.net/huagong_adu/article/details/7937616
	+ https://blog.csdn.net/v_july_v/article/details/41209515
	+ LDA，就是将原来向量空间的词的维度转变为Topic的维度，这一点是十分有意义的

+ others
![](/home/apollo/Pictures/MF5.png)