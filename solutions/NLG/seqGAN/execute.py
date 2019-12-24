import os
import tensorflow as tf
import numpy as np
import sys
import time
import gen.generator as gens
import disc.discriminator as disc
import random
import utils.conf as conf
import utils.data_utils as data_utils
from six.moves import xrange


gen_config = conf.gen_config
disc_config = conf.disc_config
evl_config = conf.disc_config

_buckets=gen_config.buckets

def __merge_data_for_disc(sess, gen_model, vocab, source_inputs, source_outputs, encoder_inputs, decoder_inputs, target_weights, bucket_id, mc_search=False):
    """
    功能：
        Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)

    参数：
        gen_model: 生成模型
        vocab : 字典，当前未使用
        source_inputs: 原始的 query token list, 大小为 [batch_size, sequence_length, vocabulary_size]
        source_outputs: 原始的 answer token list, 大小为 [batch_size, sequence_length, vocabulary_size] 
        encoder_inputs: gen_model encoder 使用的数据， 
        decoder_inputs: gen_model decoder 使用的数据
        target_weights:  暂时不了解 ？？？
        bucket_id: 选择的 bucket
        mc_search: 是否进行 蒙特卡洛 搜索

    返回：
        合并后的 train_query, train_answer, train_labels， 大小为 （1 + 1） * batch_size 或者 （1 + beam_size) * batch_size
        
        两者的 train query 相同
        两者的 train answer 不同， 前 batch_size 个 为 source 真实数据， 后 batch size 个 为 gen model 生成的数据
        两者的 train lable 不同， 前 batch_size 个 为 1， 后 batch size 个 为 0
        
    """
    train_query, train_answer = [], []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]
    """
     获得 原始 数据的 query， answer， label
     label 为1 
    """
    for query, answer in zip(source_inputs, source_outputs): 
        """
        query 和 answer 倒转加padding
        """
        query = query[:query_len] + [int(data_utils.PAD_ID)] * (query_len - len(query) if query_len > len(query) else 0)
        train_query.append(query)
        answer = answer[:-1] # del tag EOS
        answer = answer[:answer_len] + [int(data_utils.PAD_ID)] * (answer_len - len(answer) if answer_len > len(answer) else 0)
        train_answer.append(answer)
        train_labels = [1 for _ in source_inputs]

    """
    根据原始的label 生成answer
    label 为 0
    """
    def decoder(num_roll):

        for _ in range(num_roll):
            # encoder_state, loss, outputs. 猜测  output_logits 大小为 [seq_len, batch_size]
            _, _, output_logits = gen_model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

            seq_tokens = []
            resps = []

            for seq in output_logits: # 遍历 所有 sequence 的 位置
                row_token = []
                for t in seq: # 遍历当前位置中的所有的词
                    row_token.append(int(np.argmax(t, axis=0))) # 找到 当前位置的 最佳 token 写入
                seq_tokens.append(row_token) # 返回当前 sequence 的 最佳解码结果

            # 格式转化 与结果处理
            seq_tokens_t = []
            for col in range(len(seq_tokens[0])): # len(seq_tokens[0]) 为 一个 sequence 的长度
                seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])  # len(seq_tokens) 

            for seq in seq_tokens_t: # seq_tokens_t 大小为 batch_size
                if data_utils.EOS_ID in seq:
                    resps.append(seq[:seq.index(data_utils.EOS_ID)][:gen_config.buckets[bucket_id][1]])
                else:
                    resps.append(seq[:gen_config.buckets[bucket_id][1]])

            # 数据 append 到 之前的  train_query, train_answer, train_labels
            for i, output in enumerate(resps): # resps 大小为 batch_size
                output = output[:answer_len] + [data_utils.PAD_ID] * (answer_len - len(output) if answer_len > len(output) else 0)
                train_query.append(train_query[i])
                train_answer.append(output)
                train_labels.append(0)

        return train_query, train_answer, train_labels

    if mc_search:
        # 进行 蒙特卡洛搜索, decoder beam_size 次
        train_query, train_answer, train_labels = decoder(gen_config.beam_size)
    else:
        # 不进行 蒙特卡洛 搜索, decode 一次
        train_query, train_answer, train_labels = decoder(1)

    return train_query, train_answer, train_labels

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob

def __get_reward_or_loss(sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=False):
    """
    功能：
        获得 reward 或者 loss
    输入：
        bucket_id :
        disc_model : 
        train_query : 
        train_answer :
        train_labels : 
        forward_only : 为 True 时 更新  Gen， 为 False 时 更新 Disc 
    """
    feed_dict={}
    for i in xrange(len(train_query)):
        feed_dict[disc_model.query[i].name] = train_query[i]

    for i in xrange(len(train_answer)):
        feed_dict[disc_model.answer[i].name] = train_answer[i]

    feed_dict[disc_model.target.name]=train_labels

    loss = 0.0
    """
    参数更新
    """
    if forward_only:  #更新G的时候设置为true，更新D的时候设置为false
        # 产生 reward
        # 更新 生成器 gen
        fetches = [disc_model.b_logits[bucket_id]]
        logits = sess.run(fetches, feed_dict)
        logits = logits[0]  
    else:
        # disc_model.b_train_op 更新 识别器的参数
        # update 识别器 disc 
        fetches = [disc_model.b_train_op[bucket_id], disc_model.b_loss[bucket_id], disc_model.b_logits[bucket_id]]
        # train_op, loss, logits = sess.run(fetches,feed_dict)
        _, loss, logits = sess.run(fetches,feed_dict)

    # softmax operation
    logits = np.transpose(softmax(np.transpose(logits)))

    reward, gen_num, real_num = 0.0, 0, 0
    for logit, label in zip(logits, train_labels):
        #只算负类的reward
        if int(label) == 0:
            reward += logit[1]  #logit[1] 表示只取负类的概率
            gen_num += 1
        else:
            real_num += 1
    reward = reward / gen_num

    # print( "生成数据的总数，gen_num is ", gen_num)
    # print( "真实数据的总数，real_num is ", real_num)

    return reward, loss

# 对抗训练模块
def al_train():
    with tf.Session() as sess:
        # 将数据处理成 不同bucket
        vocab, rev_vocab, dev_set, train_set = gens.prepare_data(gen_config)
        for set in train_set:
            print("al train len: ", len(set))
        # bucket 信息
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes)) # 所有bucket的大小
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))] # 各个 bucket 所占的比例 [0.1, 0.34, 0.67, 0.81, 1.0]
        # 创建模型
        disc_model = disc.create_model(sess, disc_config, disc_config.name_model)
        gen_model = gens.create_model(sess, gen_config, forward_only=False, name_scope="genModel") # 默认的 forward_only 是 false

        current_step = 0
        step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0


        while True:
            current_step += 1
            start_time = time.time()
            random_number_01 = np.random.random_sample() # 返回一个 0到1 之间的数
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01]) # 找到大于 random_number_01 的那个最小的 bucket_id

            #
            # print("==================Updating Discriminator: %d=====================" % current_step)
            #

            # 1.Sample (X,Y) from real disc_data
            encoder_inputs, decoder_inputs, target_weights, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, gen_config.batch_size) # 获得所有batch 之后的数据

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_query, train_answer, train_labels = __merge_data_for_disc(sess, gen_model, vocab, source_inputs, source_outputs, encoder_inputs, decoder_inputs, target_weights, bucket_id, mc_search=False)
            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)

            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            _, disc_step_loss = __get_reward_or_loss(sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=False)
            disc_loss += disc_step_loss / disc_config.steps_per_checkpoint

            # 
            # print("==================Updating Generator: %d=========================" % current_step)
            #

            # 1.Sample (X,Y) from real disc_data
            encoder, decoder, weights, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, gen_config.batch_size)

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
            train_query, train_answer, train_labels = __merge_data_for_disc(sess, gen_model, vocab, source_inputs, source_outputs, encoder, decoder, weights, bucket_id, mc_search=True)
            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)

            # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            
            reward, _ = __get_reward_or_loss(sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=True)
            reward = reward - 0.5
            batch_reward += reward / gen_config.steps_per_checkpoint

            # 4.Update G on (X, ^Y ) using reward r   #用poliy gradient更新G
            gan_adjusted_loss, gen_step_loss, _ =gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False, reward=reward, up_reward=True, debug=True)
            gen_loss += gen_step_loss / gen_config.steps_per_checkpoint

            # 5.Teacher-Forcing: Update G on (X, Y )   #用极大似然法更新G
            t_adjusted_loss, t_step_loss, a = gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False)
            t_loss += t_step_loss / gen_config.steps_per_checkpoint
           
            if current_step % gen_config.steps_per_checkpoint == 0:

                for i in xrange(3): # 输出3条样本进行查看
                    print("tmp query is ", "".join([tf.compat.as_str(rev_vocab[output]) for output in train_query[i]])) # 打印当前的query 
                    print("label: ", train_labels[i]) # 打印ground truth label 1
                    print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in train_answer[i] if output != 0]))

                    for idx in range(1, gen_config.beam_size):
                        print('i', i, "idx", idx, 'beam_size', gen_config.beam_size, 'tmp', idx*gen_config.batch_size + i)
                        print("label: ", train_labels[ idx * gen_config.batch_size + i],  " text is ", "".join([tf.compat.as_str(rev_vocab[output]) for output in train_answer[ idx * gen_config.batch_size + i] if output != 0]))
 

                step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint

                print("*" * 20 + " show results " + "*" * 20)
                print("current_steps: %d, step time: %.4f, disc_loss: %.3f, gen_loss: %.3f, t_loss: %.3f, reward: %.3f"
                      %(current_step, step_time, disc_loss, gen_loss, t_loss, batch_reward))

                print("current_steps: %d, save disc model" % current_step)
                disc_ckpt_dir = os.path.abspath(os.path.join(disc_config.train_dir, "checkpoints"))
                if not os.path.exists(disc_ckpt_dir):
                    os.makedirs(disc_ckpt_dir)
                disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                disc_model.saver.save(sess, disc_model_path, global_step=disc_model.global_step)

                print("current_steps: %d, save gen model" % current_step)
                gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                if not os.path.exists(gen_ckpt_dir):
                    os.makedirs(gen_ckpt_dir)
                gen_model_path = os.path.join(gen_ckpt_dir, "gen.model")
                gen_model.saver.save(sess, gen_model_path, global_step=gen_model.global_step)

                step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
                sys.stdout.flush()

def main(_):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # step_1 training gen model
    # gens.pretrain(gen_config)

    # step_2 gen training data for disc
    # gens.decoder(gen_config)

    # step_3 training disc model
    # disc.hier_train(disc_config, evl_config)

    # step_4 training al model
    al_train()


if __name__ == "__main__":

    tf.app.run()
