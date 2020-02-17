import tensorflow as tf
import pickle
import utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import Evaluate
from es_tool import *


model_path = './model/model.40'

mode_train = True
mode_load = False
mode_predict = False
mode_debug = False

hidden_unit = 512 
sequence_len = 32
learning_rate = 0.01


if mode_debug == True:
    data_file = 'history-true-false-top10000.pkl'
    batch_size = 64
else:
    batch_size = 1024
    data_file = 'history-true-false.pkl'

"""
 history 结构
 [
 ["今天天气不错啊","是的，天气好心情就好","有没有适合这个天气的户外活动？"],
  ...
 [""],
 [""]
  ]

 true_utt 结构 
 [
 ["晴朗的天气适合户外跑步"],
  ...
 [""],
 [""]
 ]

 actions 结构 选一个随机负样本， 当前为随机生成（考虑用nlg）(优化点1)
 [
  ['下雨天适合在家里呆着‘]
  []
  []
 ]

"""


emb_file = open('embedding_matrix.pkl', 'rb')
emb = pickle.load(emb_file)
emb_file.close()
embeddings = emb['embedding_matrix']

if mode_debug == True:
    print('加载embedding')
    print(embeddings[0])
    print(embeddings[1])
    print(embeddings[2])

vocab_file = open('/export/home/sunhongchao1/Prototype-Robot/corpus/char2idx_tencent.pkl', 'rb')
char2idx = pickle.load(vocab_file)
word2idx = char2idx
idx2word = {}

for (char, idx)in word2idx.items():
    idx2word[idx] = char
if mode_debug == True:
    print('加载字典')
    print('的 : index', word2idx['的'])
    print('你 : index', word2idx['你'])
    print('1 : word', idx2word[1])
    print('2 : word', idx2word[2])

class SCN():
    def __init__(self):
        self.max_num_utterance = 2 # 上下文最大轮数
        self.negative_samples = 1 # 负样本个数
        self.max_sentence_len = sequence_len # 文本最大长度
        self.word_embedding_size = 200 # 需要改
        self.rnn_units = hidden_unit
        self.total_words = 22752
        self.batch_size = batch_size
        self.epoch = 100

    def LoadModel(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, model_path)
        return sess

    def BuildModel(self):

        # 上下文 placeholder
        self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance, self.max_sentence_len))
        # response placeholder
        self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        # 标记是正确结果 或者 错误结果
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        # embedding 初始化
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
        # response 长度
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        # 所有 utterance 的长度
        self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.max_num_utterance))

        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.word_embedding_size), dtype=tf.float32, trainable=False)
        self.embedding_init = word_embeddings.assign(self.embedding_ph) # 将placeholder 赋值给 value，run 的时候给placeholder 赋值时 同事更新 value
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph) # 使用lookup 将 utterance_ph 进行 embedding 表示
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph) # 使用lookup 将 response_ph 进行 embedding 表示

        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer()) # 初始化GRU 单元， 用于句子编码
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1) # 改变信息表示方式
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1) # 改变信息表示方式
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer()) # 最终GRU 单元，用于结果输出
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings, sequence_length=self.response_len, dtype=tf.float32, scope='sentence_GRU') # 返回值 (outputs， states)， 获得response的GRU表示
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings) # 计算utterance 和 response 的匹配矩阵
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings, sequence_length=utterance_len, dtype=tf.float32, scope='sentence_GRU') # 获得每个utterance 的GRU 表示
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID', kernel_initializer=tf.contrib.keras.initializers.he_normal(), activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3), padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major

        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):  
            self.logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
            self.y_pred = tf.nn.softmax(self.logits, name = 'preds')
            self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true,
                                                                      logits=self.logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.total_loss)

    def Predict(self, sess, history, es_candidate_list):
        """
        根据 对话历史(当前知识单轮) 和 es 粗排结果进行精确排
        sess
        single
        """

        history_index = []
        for item in history:
            history_index.append([ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in item])

        candidate_index = []
        for item in es_candidate_list:
            candidate_index.append([ word2idx[tmp] if tmp in word2idx.keys() else 0 for tmp in item])

        history = [history_index] * len(candidate_index)
        true_utt = candidate_index
        
        if mode_debug == True:
            print('history embedding', history)
            print('true_utt embedding', true_utt)

        self.all_candidate_scores = []
        history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
        history, history_len = np.array(history), np.array(history_len)
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))

        if mode_debug == True:
            print('history embedding padding', history)
            print('true_utt embedding padding', true_utt)
            print('history len', history_len)
            print('true utt len', true_utt_len)

        low = 0
        while True:
            feed_dict = {self.utterance_ph: np.concatenate([history[low:low + self.batch_size]], axis=0),
                         self.all_utterance_len_ph: np.concatenate([history_len[low:low + self.batch_size]], axis=0),
                         self.response_ph: np.concatenate([true_utt[low:low + self.batch_size]], axis=0),
                         self.response_len: np.concatenate([true_utt_len[low:low + batch_size]], axis=0),
                         }
            candidate_scores, logits = sess.run([self.y_pred, self.logits], feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1]) # 匹配 
            if mode_debug == True :
                print('# logits', logits)
                print('# candidate socres', candidate_scores)
                print('all andidate socres', self.all_candidate_scores)
            low = low + self.batch_size
            if low >= history.shape[0]:
                break
        all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)

        return all_candidate_scores


    # def Evaluate(self,sess):
        # with open(evaluate_file, 'rb') as f:
        #    history, true_utt,labels = pickle.load(f)
        # self.all_candidate_scores = []
        # history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
        # history, history_len = np.array(history), np.array(history_len)
        # true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        # true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
        # low = 0
        # while True:
        #     feed_dict = {self.utterance_ph: np.concatenate([history[low:low + 200]], axis=0),
        #                  self.all_utterance_len_ph: np.concatenate([history_len[low:low + 200]], axis=0),
        #                  self.response_ph: np.concatenate([true_utt[low:low + 200]], axis=0),
        #                  self.response_len: np.concatenate([true_utt_len[low:low + 200]], axis=0),
        #                  }
        #     candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
        #     self.all_candidate_scores.append(candidate_scores[:, 1])
        #     low = low + 200
        #     if low >= history.shape[0]:
        #         break
        # all_candidate_scores = np.concatenate(self.all_candidate_scores, axis=0)
        # Evaluate.ComputeR10_1(all_candidate_scores,labels)
        # Evaluate.ComputeR2_1(all_candidate_scores,labels)

    def TrainModel(self, countinue_train = False, previous_modelpath = "model"):
        """
        读取数据
        """
        with open(data_file, mode='rb') as f:
            results = pickle.load(f)
        print('read data done')
        history, true_utt, false_utt = results['history'], results['true_utt'], results['false_utt']
        history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
        false_utt_len = np.array(utils.get_sequences_length(false_utt, maxlen=self.max_sentence_len))
        false_utt = np.array(pad_sequences(false_utt, padding='post', maxlen=self.max_sentence_len))
        history, history_len = np.array(history), np.array(history_len)

        print('data index and padding done')

        print('history top 10', history[:10])
        print('true top 10 ', true_utt[:10])
        print('false top 10', false_utt[:10])
        print('his len top 10', history_len[:10])
        print('true len top 10', true_utt_len[:10])
        print('false len top 10', false_utt_len[:10])
        
        import random

        randnum = random.randint(0,100)
        random.seed(randnum)
        random.shuffle(history)
        random.seed(randnum)
        random.shuffle(history_len)
        random.seed(randnum)
        random.shuffle(true_utt)
        random.seed(randnum)
        random.shuffle(true_utt_len)
        random.seed(randnum)
        random.shuffle(false_utt)
        random.seed(randnum)
        random.shuffle(false_utt_len)


        """
        参数初始化
        """
        low = 0
        epoch = 1
        all_samples_len = len(true_utt)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
            else:
                saver.restore(sess,previous_modelpath)

            while epoch < self.epoch:
                n_sample = min(low + self.batch_size, history.shape[0]) - low
                negative_indices = [np.random.randint(0, false_utt.shape[0], n_sample) for _ in range(self.negative_samples)] 

                negs = np.asarray([false_utt[negative_indices[i]] for i in
                                   range(self.negative_samples)])
                negs_len = np.asarray([false_utt_len[negative_indices[i]] for i in
                            range(self.negative_samples)])

                negs = np.reshape(negs, (n_sample, -1))
                negs_len = np.reshape(negs_len, (n_sample))

                #print('negs shape', negs.shape)
                #print('negs len shape', negs_len.shape)
                #print('true utt shape', true_utt.shape)

                feed_dict = {self.utterance_ph: np.concatenate([history[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.all_utterance_len_ph: np.concatenate([history_len[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.response_ph: np.concatenate([true_utt[low:low + n_sample], negs], axis=0),
                             self.response_len: np.concatenate([true_utt_len[low:low + n_sample], negs_len], axis=0),
                             self.y_true: np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * self.negative_samples, axis=0)
                             }
            

                _, logits, loss, prob = sess.run([self.train_op, self.logits,
                                                  self.total_loss, self.y_pred], feed_dict=feed_dict)
                low += n_sample
                if low >= history.shape[0]:
                    print(" ######## epoch:{}, 进度:{:.2f}%, total_loss:{:.4f}".format(epoch,
                                                                         low/all_samples_len*100,
                                                                         loss))
                    print(logits[:4])
                    print(prob[:4])
                    low = 0
                    saver.save(sess,"model/model.{0}".format(epoch))
                    epoch += 1

                    """
                    save pb model
                    """
                    pb_dir = './model'

                    from tensorflow.python.framework import graph_util
                    # print(sess)
                    # print(sess.graph_def)
                    trained_graph = graph_util.convert_variables_to_constants(sess,
                                                              sess.graph_def,
                                                              output_node_names=['logits/preds'])
                    tf.train.write_graph(trained_graph, pb_dir, 'model-' + str(epoch) + ".pb", as_text=False)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    scn =SCN()
    scn.BuildModel()

    if mode_train: 
        """
        进行模型训练
        """
        scn.TrainModel()
    elif mode_load:  
        """
        加载模型，单条输入，并进行结果输出
        """
        sess = scn.LoadModel()
        search = Search()
        search.create_index()
        obj = ElasticObj('qa_info', 'qa_detail')
        history = sys.argv[1:] 
        query = sys.argv[-1]
        print('history', history)
        print('input', query)
        answer_list = obj.Get_Data_By_Body(query)
        # answer_list = list(set(answer_list))
        result = scn.Predict(sess, query, answer_list)
        exp_result = np.exp(result)
        result = [item/np.sum(exp_result) for item in exp_result]
        result_str = [str(round(tmp,3)) for tmp in result ]
        print('smn score', ' '.join(result_str))    
        best_idx = np.argmax(result)
        print("best answer", answer_list[best_idx], 'best idx', best_idx)
    elif mode_predict:
        """
        加载模型，批量输入，并进行结果输出
        """
        sess = scn.LoadModel()
        
        """
        导入ES 库
        """
        obj = ElasticObj('new_qa_name', 'new_qa_type')
        #query_list = ['你好啊','会打篮球吗', '喜欢足球吗', '姚明', '物理学',
        #             '武磊的足球踢的好吗', '你和小冰谁厉害', '许褚']

        query_list = ['你好啊 ', '你会唱歌吗 ', '你吃饭了吗 ', '你会跳舞吗 ',
                      '你是哪里人 ', '窗前明月关 ', '你知道航母吗 ',
                      '我会打篮球 ', '我打架可厉害了 ', '你有钱吗 ',
                      '大河向东流 ', '消灭人类暴政 ', '我要做海贼王的男人 ',
                      '钢铁侠变身 ', '那只竹鼠中暑了 ', '天王盖地虎 ',
                      '说个笑话 ', '北京天气 ', '给大爷笑一个', '你知道乔丹吗', '你知道周杰伦吗 ', '苹果手机不错啊 ', '华为手机不错啊',
                      '小米手机真垃圾 ', 'oppo手机不错啊 ',
                      '西门子冰箱不错啊 ', '苹果手机太贵了 ', '华为手机真耐用', '小米手机性价比高 ', 'oppo手机不错啊 ',
                      '西门子冰箱不错啊 ', '你喜欢航模吗 ', '飞机模型好玩吗 ',
                     '本田的车太小了 ', '牧马人简直太帅了 ', '张辽威震逍遥津 ',
                     '许褚裸衣战马超 ', '西游记的故事 ', '水浒传的故事 ',
                      '白娘子的传说 ', '三国演艺的故事 ', '何不食肉糜 ',
                      '八王之乱 ', '五胡十六国 ', '蜡笔小新 ', '派大星 ',
                      '你养狗吗 ', '自然语言处理 ', '最近都听啥歌了 ',
                      '吃太饱了怎么办 ', '如何减肥 ', '出门跑步吧 ',
                      '跑个几公里 ', '做完俯卧撑，再来几个仰卧起坐 ',
                      '刷火锅好吃吗 ', '羊肉怎么吃好吃 ',
                      '烤羊腿，羊蝎子，羊肉火锅 ', '出门去玩了 ',
                      '北京堵车好厉害啊 ', '好的，不着急哈', '啥时候放假呀',
                      '真无聊', '周末出去玩不', '周一日常虚弱',
                      '好好休息一下把', '下午还有计划去健身房吗', '我感冒了待会我自己吃', '还需要补充什么？', '你就是不想上班',
                      '我热爱工作',
                      '哈哈，那你今天陪我加班，我就认为你热爱工作',
                      '我每天都加班啊', '昨天我加班，没看到你',
                      '损失了好几个亿了', '嗯嗯，没事没事', '今天天气真好',
                      '外面下雪了', '窗外有只松鼠', '我是一只鱼', '我在成都',
                      '你和小冰谁更厉害', '我要喝牛奶', '今天好冷啊',
                      '不如跳舞', '你是男生还是女生', '你最近在听什么音乐',
                      '马上要过年啦', '早上好！', '再见！', '晚安！', '谢谢！',
                     '太好了！多亏你帮忙！', '太好了！', '太奇妙了！',
                      '多美妙啊', '你学得真快！', '你真棒！', '你真能干！',
                      '我真为你高兴！', '我好羡慕你！', '我来帮助你好吗？',
                      '请你帮帮我！', '你真会思考！', '你真能想办法l',
                      '我们都按游戏规则玩，好吗？', '不怕，让我们来想想办法！',
                     '有进步，再试一次！', '再来一次！', '你真有理想！ ',
                      '和你在一起，真快乐！', '你猜我想问啥？',
                      '你猜我猜不猜？', '瞅啥呢？', '浪来了，浪来了。',
                      '浪来了也没有他。', '说这个有什么用呢', '送我一台手机吧',
                     '知道我鞋子穿几号吗', '妈妈称赞我数学考60分',
                      '今天又午睡到三点', '周末想去哪玩', '巧克力真好吃',
                      '唱首有关大海的歌', '你知道小明的笑话吗',
                      '20元一餐可以吃什么', '肚子疼怎么办', '先有鸡还先有蛋',
                      '我是一只小小鸟', '这个香瓜有点老', '云从龙，风从虎',
                      '邻家妹妹可有主', '看山不是山，看水不是水',
                      '这个龙虾挺辣嘴', '一路向北，可曾后悔', '考试可以抱大腿',
                     '最近怎么样', '吃了吗', '你瞅啥', '啥时候换手机',
                      '今天天气不错啊  ', '穿秋裤了吗', '晚上去哪吃', '你叫啥',
                     '你猜我叫啥', '那你为什么这么说啊', '你刚刚在想什么',
        '你有梦想吗？', '想不想我？', '你周五怎么没有回我电话啊',
        '你知道我是怎么想的吗', '你知道我的信仰是什么吗', '我的未来不是梦',
        '眼下就是要步步为营，不容半点闪失']
        
        def find_error(input_list, input_str):
            for item in input_list:
                if item in input_str:
                    return True
            return False

        for query in query_list:
            print("#" * 30)
            print('query is ', query)
            es_list = obj.Get_Data_By_Body(query)

            es_list = list(set(es_list))
            es_list.sort(key = lambda i:len(i),reverse=True)  
            result = scn.Predict(sess, [query], es_list)

            exp_result = np.exp(result)
            result = [item/np.sum(exp_result) for item in exp_result]

            result_str = [str(round(tmp,3))  for tmp in result]
            
            print('es results with smn score ')
            count = 0 
            for text, score in zip(es_list, result_str):
                print( 'idx :', count, ' score is :', score, ' text is :', text)
                count = count + 1
            best_idx = np.argmax(result)
            print("best answer", es_list[best_idx], 'best idx', best_idx)


    else:
        pass

# check_list=['撸','事件','成都','北京','上海','那个','这个','图片','晚安', '早安', '上午好','下午好','晚上好', '传说中','吃吃吃','礼物','下台','转一个','转运','政绩工程','朱艳艳','毛泽东','做客','记录','Nick','听众','中国梦','博鳌','求救','达人秀','演唱会','明星','PM','北京空气','早上','清早','凌晨','上午','中午','晚上','今晚','昨天','明天','后天','销量','到货','si','鸡鸡','小通','旺财','直播','庆祝','销量','视频','mv','#','#', '-','<','>','《','》', "@", '【', '】', '？', ' ——', '_', '"', "'",':','：', '‘','’']
#            new_answer_list = []
#            check_list = ['旺财', '小通', '鸡鸡']
#            for answer in answer_list:
#                if find_error(check_list, answer):
#                    pass
#                else:
#                    new_answer_list.append(answer)
