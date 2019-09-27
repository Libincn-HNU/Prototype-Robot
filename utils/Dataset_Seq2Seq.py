from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Dataset_Seq2seq(object):
    """
    batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    """
    def __init__(self, data_path, tokenizer, max_seq_len):

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.word2idx = tokenizer.word2idx
        self.max_seq_len = max_seq_len

    def __pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        """
        :param sequence:
        :param maxlen:
        :param dtype:
        :param padding:
        :param truncating:
        :param value:
        :return: sequence after padding and truncate
        """
        x = (np.ones(maxlen) * value).astype(dtype)

        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
            trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def __encode_text_sequence(self, text, do_padding, do_reverse, do_mask):
        """
        :param text:
        :return: convert text to numberical digital features with max length, paddding
        and truncating
        """
        words = list(text)
        words.append('<END>')
        words.reverse()
        words.append('<BEG>')
        words.reverse()

        sequence = [self.word2idx[w] if w in self.word2idx else self.word2idx['<UNK>'] for w in words]
        mask = [self.word2idx['<PAD>'] * self.max_seq_len]

        if len(sequence) == 0:
            sequence = [0]

        if do_reverse:
            sequence = sequence[::-1]

        if do_padding:
            sequence = self.__pad_and_truncate(sequence, self.max_seq_len, value=0)

        if do_mask:
            for index in range(len(words)):
                mask[index] = 1

        return sequence, mask

    def preprocess(self):

        fin = open(self.data_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        querys = []
        responses = []

        """
        read data from corpus
        """
        for line in lines:
            cut_list = line.split('\t')

            if len(cut_list) == 2:
                querys.append(cut_list[0])
                responses.append(cut_list[1])
            else:
                raise Exception("Raise Exception")

        encoder_list = []
        for text in querys:
            tmp, mask = self.__encode_text_sequence(text, True, False, False)
            encoder_list.append(tmp)

        decoder_list = []
        target_list = []
        mask_list = []

        """
        反序可能提高模型效果
        """

        for text in responses:
            tmp, mask = self.__encode_text_sequence(text, True, False, True)
            decoder_list.append(tmp)
            target_list.append(tmp[1:])
            mask_list.append(mask)


def build_dataset(data_path, tokenizer, max_seq_len, batch_size):
    dataset = Dataset_Seq2seq(corpus=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)

    data_loader = tf.data.Dataset.from_tensor_slices({'encoder': dataset.encoder_list, 'decoder': dataset.label_list, 'target': dataset.target_list, 'mask': dataset.mask_list }).batch(batch_size)

    return data_loader





