### End2End

- https://blog.csdn.net/sinat_33761963/article/details/79011160(最后)

### End2End Task Oriented Dialog System

- Learning End-to-End Goal-Oriented Dialog
  - Antoine Bordes and Jason Weston. Learning end-to-end goal-oriented dialog. arXiv preprint arXiv:1605.07683, 2016
  - 基于Memory Network在QA chatbot领域充分展示了优势，而Paper Learning End-to-End Goal-Oriented Dialog则将其应用在了领域任务型的对话系统(Task Oriented Dialog System)中。模型还是原来的模型，只是用法不同而已
  - Memory Network其实验数据为以餐馆预订为目的的bAbI数据集，已有开源的[数据和代码](http://link.zhihu.com/?target=https%3A//github.com/vyraun/chatbot-MemN2N-tensorflow)
  - 总结一下其优缺点：应用Memory Network的优势在于不用独立涉及 SLU,DST，也就是直接的端到端的一种训练，在其缺点在于很难结合知识库。同时在该论文中的实验是基于模板检索的各种模板还是需要自己制定，也就是存在迁移差的问题
- LSTM-based dialog & Hybird Code Networks
  - Jason D Williams and Geoffrey Zweig. End-to-end lstm-based dialog control optimized with supervised and reinforcement learning. arXiv preprint arXiv:1606.01269, 2016.
  - Williams J D, Asadi K, Zweig G. Hybrid Code Networks: practical and efficient end-to-en dialog control with supervised and reinforcement learning[J]. arXiv preprint arXiv:1702.03274, 201
  - 两篇论文较为相似
  - 首先在motivation上，可以说都是客服成长过程，这个在 第一篇 开头有描述，在训练的时候可以看作是专家在训练一个新入职的客服，由专家提供一系列example dialogues，和一些标准答案，然后让模型去学习。学成之后上线服务，在服务的同时有结合反馈来继续学习。然后是在模型方面，从图中就可以看出相似之处，整体流程方面大概是首先输入会话，然后提取entity ，加上原来对话，转成向量，输入给RNN，用rnn来做DM结合一些领域知识，一些action mask和templates，输出决策，再基于模板生成对话，这样一直循环。最后是在训练方法上面，都提出了SL和RL都可以独立训练或者结合来训练的方式
  - 在网上找到Hybrid Code Networks模型的实验代码--[代码和数据](http://link.zhihu.com/?target=https%3A//github.com/johndpope/hcn)，实验数据还是以餐馆预订为目的的bAbI数据集，可以和上面的Memory Network base 模型做下对比实验
- **A network Based end To End trainable task Oriented dialogue system**
  - Wen T H, Vandyke D, Mrksic N, et al. A network-based end-to-end trainable task-oriented dialogue system[J]. arXiv preprint arXiv:1604.04562, 2016
- End To End task Completion neural dialogue systems
  - 总结其优缺点：其优点很明显就是在训练的时候是端到端的，同时User Simulator在一定程度上解决了end to end Task oriented dialogue systems训练数据难获得的问题。然后其劣势和挑战在于首先数据集不大，在训练的时候也较依赖于其特定的数据集
  - 这篇paper开源了其[代码和数据](http://link.zhihu.com/?target=https%3A//github.com/MiuLab/TC-Bot)，是纯python无框架写的
  - A user simulator for task-completion dialogues
- IDEA of End2End
  - 首先是在end to end 模型中，其实就是在想怎么把之前做的分模块要分开训练的多个模型有机的结合统一成一个新的模型，比较常见的是比如DM用RNN来做，更好的跟踪状态，用基于模板或者基于生成式或者两者结合的方式来生成对话等
  - 在训练方式方面，RL算法在上面的论文用普遍都有用，也得到了非常不错的效果，这个是个很不错的方式，也是可以继续提出创新点的地方
  - 最后在task oriented dialogue systems方面end to end的数据缺乏，在[7]提出的User Simulator 是一个比较不错的方式来模拟用户训练，当然在发展end to end Task oriented dialogue systems中，数据集还是个很大的问题
  - Next
    - 用更少的领域先验知识来构建模型
    - 怎么才能提高其领域迁移性的问题