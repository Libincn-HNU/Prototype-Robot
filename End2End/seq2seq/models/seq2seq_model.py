import tensorflow as tf
from seq2seq import embedding_attention_seq2seq


class Seq2SeqModel():

    def __init__(self, args, tokenzier):

        # source_vocab_size, target_vocab_size, en_de_seq_len, hidden_size, num_layers,
        # batch_size, learning_rate, num_samples = 1024,
        # forward_only = False, beam_search = True, beam_size = 10

        self.source_vocab_size = len(tokenzier.word2idx)
        self.target_vocab_size = len(tokenzier.word2idx)

        self.hidden_size = self.hidden_dim
        self.num_layers = 3
        self.batch_size = self.batch_size
        self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_drop = tf.placeholder(tf.float32)

        self.forward_only = True
        self.beam_search = True
        self.beam_size = 3
        self.num_sampled = 24

        self.output_projection = None

    def create_rnn_cell(self):
        """
        定义encoder和decoder阶段的多层dropout RNNCell
        :return:
        """
        encoDecoCell = tf.contrib.rnn.BasicLSTMCell(self. hidden_size)
        encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=self.keep_drop)

        return encoDecoCell

    def sample_loss(self, logits, labels):
        """
        调用sampled_softmax_loss函数计算sample loss，这样可以节省计算时间
        :param logits:
        :param labels:
        :return:
        """
        # output projection
        w = tf.get_variable('proj_w', [self.hidden_size, self.target_vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable('proj_b', [self.target_vocab_size])
        self.output_projection = (w, b)

        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, labels=labels, inputs=logits, num_sampled=self.num_sampled, num_classes=self.target_vocab_size)

    def _process(self):

        """
        定义输入的placeholder，采用了列表的形式
        """
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.target_weights = []

        for i in range(self.max_seq_len):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None, ], name="encoder{0}".format(i)))
        for i in range(self.max_seq_len):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None, ], name="decoder{0}".format(i)))
            self.decoder_targets.append(tf.placeholder(tf.int32, shape=[None, ], name="target{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None, ], name="weight{0}".format(i)))

        encoCell = tf.contrib.rnn.MultiRNNCell([self.create_rnn_cell() for _ in range(self.num_layers)])

        if self.forward_only:
            """
            test模式，将上一时刻输出当做下一时刻输入传入
            """
            if self.beam_search:
                """
                beam_search
                """
                self.beam_outputs, _, self.beam_path, self.beam_symbol = embedding_attention_seq2seq(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    encoCell,
                    num_encoder_symbols=self.source_vocab_size,
                    num_decoder_symbols=self.target_vocab_size,
                    embedding_size=self.hidden_size,
                    output_projection=self.output_projection,
                    feed_previous=True)
            else:
                """
                greedy search
                """
                decoder_outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    encoCell,
                    num_encoder_symbols=self.source_vocab_size,
                    num_decoder_symbols=self.target_vocab_size,
                    embedding_size=self.hidden_size,
                    output_projection=self.output_projection,
                    feed_previous=True)

                if self.output_projection is not None:
                    # 因为seq2seq模型中未指定output_projection，所以需要在输出之后自己进行output_projection
                    self.outputs = tf.matmul(decoder_outputs, self.output_projection[0]) + self.output_projection[1]
        else:
            """
            train 模式
            因为不需要将output作为下一时刻的输入，所以不用output_projection
            """
            decoder_outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs,
                self.decoder_inputs,
                encoCell,
                num_encoder_symbols=self.source_vocab_size,
                num_decoder_symbols=self.target_vocab_size,
                embedding_size=self.hidden_size,
                feed_previous=False)
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                decoder_outputs,
                self.decoder_targets,
                self.target_weights,
                softmax_loss_function=self.sample_loss())

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
            self.trainer = opt.minimize(self.loss)

        self.saver = tf.train.Saver(tf.all_variables())

    def run(self, session, encoder_inputs, decoder_inputs, decoder_targets, target_weights, go_token_id):
        feed_dict = {}
        if not self.forward_only:
            feed_dict[self.keep_drop] = 0.5
            for i in range(self.en_de_seq_len[0]):
                feed_dict[self.encoder_inputs[i].name] = encoder_inputs[i]
            for i in range(self.en_de_seq_len[1]):
                feed_dict[self.decoder_inputs[i].name] = decoder_inputs[i]
                feed_dict[self.decoder_targets[i].name] = decoder_targets[i]
                feed_dict[self.target_weights[i].name] = target_weights[i]
            run_ops = [self.optOp, self.loss]
        else:
            feed_dict[self.keep_drop] = 1.0
            for i in range(self.en_de_seq_len[0]):
                feed_dict[self.encoder_inputs[i].name] = encoder_inputs[i]
            feed_dict[self.decoder_inputs[0].name] = [go_token_id]
            if self.beam_search:
                run_ops = [self.beam_path, self.beam_symbol]
            else:
                run_ops = [self.outputs]

        """
        outputs
        """
        outputs = session.run(run_ops, feed_dict)
        if not self.forward_only:
            return None, outputs[1]
        else:
            if self.beam_search:
                return outputs[0], outputs[1]