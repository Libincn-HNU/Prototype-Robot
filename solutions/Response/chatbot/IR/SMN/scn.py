import tensorflow as tf
import pickle
import utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import Evaluate
from es_tool import *


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

            actions 结构 选一个随机负样本， 当前为随机生成（考虑用nlg）
            [
                ['下雨天适合在家里呆着‘]
                []
                []
            ]

"""


if False:
    print('加载已经处理好的数据')
    load_file = open(data_file,"rb")
    results = pickle.load(load_file)
    load_file.close()

print('加载embedding')
emb_file = open('embedding_matrix.pkl', 'rb')
emb = pickle.load(emb_file)
emb_file.close()
embeddings = emb['embedding_matrix']

print('加载字典')
vocab = open('/export/home/sunhongchao1/Prototype-Robot/solutions/FAQ/NLG/seqGAN/gen_data/vocab5000.all', 'r', encoding='utf-8', newline='\n', errors='ignore')
word2idx = {}
idx2word = {}

for idx, value in enumerate(vocab):
    word2idx[value.strip()] = idx
    idx2word[idx] = value.strip()

class SCN():
    def __init__(self):
        self.max_num_utterance = 10 # 上下文最大轮数
        self.negative_samples = 1 # 负样本个数
        self.max_sentence_len = 50 # 文本最大长度
        self.word_embedding_size = 200 # 需要改
        self.rnn_units = 200 
        self.total_words = 5000
        self.batch_size = 1024

    def LoadModel(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess,"./model/model.4")
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
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        tf.summary.scalar('loss', self.total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.total_loss)

    def Predict(self, sess, single_history, true_utt_list):
        
        tmp_single_history = []
        for tmp_history in single_history:
            tmp_single_history.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in tmp_history])

        tmp_true_utt_list = []
        for tmp_utt in true_utt_list:
            tmp_true_utt_list.append([ word2idx[tmp] if tmp in word2idx.keys() else word2idx['_UNK'] for tmp in tmp_utt])

        history = [tmp_single_history] * len(true_utt_list)
        true_utt = tmp_true_utt_list

        self.all_candidate_scores = []
        history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
        history, history_len = np.array(history), np.array(history_len)
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
        low = 0
        while True:
            feed_dict = {self.utterance_ph: np.concatenate([history[low:low + 200]], axis=0),
                         self.all_utterance_len_ph: np.concatenate([history_len[low:low + 200]], axis=0),
                         self.response_ph: np.concatenate([true_utt[low:low + 200]], axis=0),
                         self.response_len: np.concatenate([true_utt_len[low:low + 200]], axis=0),
                         }
            candidate_scores = sess.run(self.y_pred, feed_dict=feed_dict)
            self.all_candidate_scores.append(candidate_scores[:, 1])
            low = low + 200
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
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("output2", sess.graph)
            train_writer = tf.summary.FileWriter('output2', sess.graph)

            with open(data_file, mode='rb') as f:
                results = pickle.load(f)

            history, true_utt, actions = results['history'], results['true_utt'], results['false_utt']

            history, history_len = utils.multi_sequences_padding(history, self.max_sentence_len)
            true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=self.max_sentence_len))
            true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=self.max_sentence_len))
            actions_len = np.array(utils.get_sequences_length(actions, maxlen=self.max_sentence_len))
            actions = np.array(pad_sequences(actions, padding='post', maxlen=self.max_sentence_len))
            history, history_len = np.array(history), np.array(history_len)
            if countinue_train == False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
            else:
                saver.restore(sess,previous_modelpath)
            low = 0
            epoch = 1
            while epoch < 30:
                n_sample = min(low + self.batch_size, history.shape[0]) - low
                negative_indices = [np.random.randint(0, actions.shape[0], n_sample) for _ in range(self.negative_samples)]
                negs = [actions[negative_indices[i], :] for i in range(self.negative_samples)]
                negs_len = [actions_len[negative_indices[i]] for i in range(self.negative_samples)]
                feed_dict = {self.utterance_ph: np.concatenate([history[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.all_utterance_len_ph: np.concatenate([history_len[low:low + n_sample]] * (self.negative_samples + 1), axis=0),
                             self.response_ph: np.concatenate([true_utt[low:low + n_sample]] + negs, axis=0),
                             self.response_len: np.concatenate([true_utt_len[low:low + n_sample]] + negs_len, axis=0),
                             self.y_true: np.concatenate([np.ones(n_sample)] + [np.zeros(n_sample)] * self.negative_samples, axis=0)
                             }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                if low % 102400 == 0:
                    print("loss",sess.run(self.total_loss, feed_dict=feed_dict))
                    # self.Evaluate(sess)
                if low >= history.shape[0]:
                    low = 0
                    saver.save(sess,"model/model.{0}".format(epoch))
                    print(sess.run(self.total_loss, feed_dict=feed_dict))
                    print('epoch={i}'.format(i=epoch))
                    epoch += 1

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    scn =SCN()
    scn.BuildModel()

    if False: 
        """
        进行模型训练
        """
        scn.TrainModel()
    elif False:  
        """
        加载模型，使用命令行参数训练
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
    else:
        """
        连同es 进行结果输出
        """

        sess = scn.LoadModel()
        
        obj = ElasticObj('chatbot_corpus_59', 'qa_detail')
        # obj = ElasticObj('qa_info', 'qa_detail')

        query_list = ['南京', '北京', '东京', '沈阳', '大连', '莫斯科', '纽约', 
                     '迪丽热巴', '古力娜扎', '穷哈', '周杰伦', '张一鸣','马斯克',
                      '五一小长假','清明','春节','元旦','中秋']
        


        def find_error(input_list, input_str):
            for item in input_list:
                if item in input_str:
                    return True
            return False

        for query in query_list:
            print("#" * 30)
            print(query)
            answer_list = obj.Get_Data_By_Body(query)
            answer_list = list(set(answer_list))
            print(answer_list)
            new_answer_list = []
            # #
            check_list = ['旺财', '小通', '鸡鸡']
            # check_list=['撸','事件','成都','北京','上海','那个','这个','图片','晚安', '早安', '上午好','下午好','晚上好', '传说中','吃吃吃','礼物','下台','转一个','转运','政绩工程','朱艳艳','毛泽东','做客','记录','Nick','听众','中国梦','博鳌','求救','达人秀','演唱会','明星','PM','北京空气','早上','清早','凌晨','上午','中午','晚上','今晚','昨天','明天','后天','销量','到货','si','鸡鸡','小通','旺财','直播','庆祝','销量','视频','mv','#','#', '-','<','>','《','》', "@", '【', '】', '？', ' ——', '_', '"', "'",':','：', '‘','’']
            for answer in answer_list:
                if find_error(check_list, answer):
                    # print('error', answer)
                    pass
                else:
                    new_answer_list.append(answer)

            result = scn.Predict(sess, [query], new_answer_list)
            # print(query, '\t', new_answer_list[np.argmax(result)])

            exp_result = np.exp(result)
            result = [item/np.sum(exp_result) for item in exp_result]

            result_str = [str(round(tmp,3)) for tmp in result ]
            print('smn score', ' '.join(result_str))    
            best_idx = np.argmax(result)
            print("best answer", answer_list[best_idx], 'best idx', best_idx)
