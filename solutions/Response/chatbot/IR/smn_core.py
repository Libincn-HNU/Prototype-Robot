
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
self.embedding_init = word_embeddings.assign(self.embedding_ph)
all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_num_utterance, axis=1)
all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.max_num_utterance, axis=1)
A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer())
reuse = None

response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings, sequence_length=self.response_len, dtype=tf.float32, scope='sentence_GRU')
self.response_embedding_save = response_GRU_embeddings
response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
matching_vectors = []
for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
    matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
    utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings, sequence_length=utterance_len, dtype=tf.float32, scope='sentence_GRU')
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

