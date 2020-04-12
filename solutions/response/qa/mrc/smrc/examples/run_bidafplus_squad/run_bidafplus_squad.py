# coding: utf-8
import sys

root_path = '/export/home/sunhongchao1/Prototype-Robot/solutions/response/qa/mrc/smrc'

sys.path.append('/export/home/sunhongchao1/Prototype-Robot/solutions/response/qa/mrc/smrc')

from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.bidafplus_squad import BiDAFPlusSQuad
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator

tf.logging.set_verbosity(tf.logging.ERROR)

data_folder = '/export/home/sunhongchao1/Prototype-Robot/corpus/mrc/dataset/squad2'
embedding_folder = '/export/home/sunhongchao1/Workspace-of-NLU/resources'
train_file = data_folder + "train-v1.1.json"
dev_file = data_folder + "dev-v1.1.json"

reader = SquadReader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

vocab = Vocabulary()
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_folder + "glove.42B.300d.txt")

train_batch_generator = BatchGenerator(vocab, train_data, batch_size=60, training=True,additional_fields=['context_word_len','question_word_len'])
eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=60,additional_fields=['context_word_len','question_word_len'])
model = BiDAFPlusSQuad(vocab, pretrained_word_embedding=word_embedding)
model.compile(tf.train.AdamOptimizer, 0.001)
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=15, eposides=2)
