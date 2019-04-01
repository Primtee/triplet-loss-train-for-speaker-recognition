# coding=utf-8

import time
import numpy as np   # 多维数据处理模块
from random import sample
import tensorflow as tf
# from dataprovider.model.vgg_model import Net
from create.data_management import load_data, mini_batch_triplet
__author__ = 'NXG'

data_set_path = 'data_set/'
model_path = 'model/model_flower'
load_model = True
saved_model_path = 'model/'
n_class_num = 100
w = 512
h = 300
c = 3
n_epoch = 250
batch_size = 32
train_class = 630


class PairwiseDistance(object):
    def __init__(self, p):
        self.norm = p

    def forward(self, a_p_batch_feature):  #
        a_feature = a_p_batch_feature[0]
        p_feature = a_p_batch_feature[1]
        for index in range(2, batch_size * 2, 2):
            a_feature = tf.concat([a_feature, a_p_batch_feature[index]], 0)
            p_feature = tf.concat([p_feature, a_p_batch_feature[index + 1]], 0)
        diff = tf.abs(a_feature - p_feature)
        diff = tf.reshape(diff, (batch_size, 1024))
        eps = 1e-4 / 1024
        out = tf.reduce_sum(pow(diff, self.norm), axis=1)  # [64 64 64]
        return tf.sqrt(out + eps)  # 1024


l2_dist = PairwiseDistance(2)


def triplet_loss(batch_feature):
    a_p = batch_feature[0: 2]  #
    a_n = tf.concat([batch_feature[0: 1], batch_feature[2: 3]], 0)  #
    for ss in range(3, batch_size * 3, 3):
        tmp_a_p = batch_feature[ss: ss + 2]
        a_p = tf.concat([a_p, tmp_a_p], 0)
        tmp_a_n = tf.concat([batch_feature[ss: ss + 1], batch_feature[ss + 2: ss + 3]], 0)
        a_n = tf.concat([a_n, tmp_a_n], 0)

    d_p = l2_dist.forward(a_p)  #
    d_n = l2_dist.forward(a_n)  #
    d_p_d_n_distance = d_n - d_p
    # compute distance
    d_p_d_n_distance = tf.clip_by_value(d_p_d_n_distance, 0.1, 1000000)
    one = tf.ones_like(d_p_d_n_distance)
    zero = tf.zeros_like(d_p_d_n_distance)
    satisfied_index = tf.where(d_p_d_n_distance <= 0.2, x=zero, y=one)
    satisfied_index_acc = tf.where(d_p_d_n_distance <= 0.2, x=one, y=zero)
    satisfied_index_ = tf.reshape(satisfied_index, shape=[tf.shape(satisfied_index)[0], 1])
    d_p_d_n_distance = 0.1 + d_p_d_n_distance  # margin
    d_p_d_n_distance = tf.expand_dims(d_p_d_n_distance, 0)
    d_p_d_n_distance_loss = tf.matmul(d_p_d_n_distance, satisfied_index_)
    # acc_compute = tf.shape(satisfied_index_)[0] / 32
    acc_compute = tf.reduce_sum(satisfied_index_acc, axis=0) / 32
    return d_p_d_n_distance_loss[0][0], acc_compute


def net(in_put, n_class=train_class, train=True, if_regularizer=True, dropout_rate=[0.6, 0.5]):
        """
        VGG
        :param train: if training
        :param if_regularizer: if re
        :param dropout_rate: fully-connection cells activation rate
        :return:
        """
        if if_regularizer:
            regularizer_method = tf.contrib.layers.l2_regularizer(0.0001)
        else:
            regularizer_method = None
        with tf.variable_scope('layer1-conv_1'):
            # input_tensor = tf.placeholder(tf.float32, shape=[None, 512, 300, 3], name='input')  # 1
            conv1_weights = tf.get_variable("weight1", [7, 7, 3, 96],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("bias1", [96], initializer=tf.constant_initializer(0.0))
            # conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 2, 2, 1], padding='SAME', name='conv1')  # 2
            conv1 = tf.nn.conv2d(in_put, conv1_weights, strides=[1, 2, 2, 1], padding='SAME', name='conv1')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases), name='relu1')
            # print('relu1:', relu1)

        with tf.name_scope("layer2_pool_1"):
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name='p1')
            # print('pool1:', pool1)
        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable("weight2", [5, 5, 96, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias22", [256], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 2, 2, 1], padding='VALID', name='conv2')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases), name='relu2')
            # print('relu2:', relu2)
        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='p2')
            # print('pool2:', pool2)
        with tf.variable_scope("layer5-conv3"):
            conv3_weights = tf.get_variable("weight3", [3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable("bias3", [256], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases), 'relu3')
            print('relu3:', relu3)
        with tf.variable_scope("layer6-conv4"):
            conv4_weights = tf.get_variable("weight4", [3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_biases = tf.get_variable("bias4", [256], initializer=tf.constant_initializer(0.0))
            conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
            relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases), 'relu4')
            print('relu4:', relu4)
        with tf.name_scope("layer7-conv5"):
            conv5_weights = tf.get_variable("weight5", [3, 3, 256, 256],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_biases = tf.get_variable("bias5", [256], initializer=tf.constant_initializer(0.0))
            conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
            relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases), 'relu5')
            print('relu5:', relu5)
        with tf.name_scope("layer8-pool3"):
            pool3 = tf.nn.max_pool(relu5, ksize=[1, 5, 3, 1], strides=[1, 3, 2, 1], padding='VALID', name='p3')
            print('pool3:', pool3)
        with tf.name_scope("layer9-conv6"):
            conv6_weights = tf.get_variable("weight6", [9, 1, 256, 4096],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer_method != None:
                tf.add_to_collection('losses', regularizer_method(conv6_weights))  # BN
            conv6_biases = tf.get_variable("bias", [4096], initializer=tf.constant_initializer(0.0))
            conv6 = tf.nn.conv2d(pool3, conv6_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv6')
            relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases), 'relu6')
            if train:
                relu6 = tf.nn.dropout(relu6, dropout_rate[0])  #
            print('relu6:', relu6)
        with tf.name_scope("layer10-pool4"):
            pool3 = tf.nn.avg_pool(relu6, ksize=[1, 1, 8, 1], strides=[1, 1, 1, 1], padding='VALID', name='p3')
            print('pool3:', pool3)

        with tf.name_scope("layer11-flatten"):
            flat = tf.contrib.layers.flatten(pool3)
            print('flatten:', pool3)
        with tf.name_scope("layer12-fullconnection"):
            ful_1 = tf.contrib.layers.fully_connected(flat, num_outputs=1024)
            ful_1 = tf.nn.relu(ful_1, name='ful_1')
            ful_1 = tf.nn.dropout(ful_1, dropout_rate[1])  #
            print('ful_1:', ful_1)
        with tf.name_scope("layer13-fullconnection_1"):
            ful_2 = tf.contrib.layers.fully_connected(ful_1, num_outputs=n_class)
            ful_2 = tf.nn.relu(ful_2, name='ful_2')
            print('ful_2:', ful_2)
        return ful_2, ful_1  # ful_1 is the speaker feature


input_holder = tf.placeholder(tf.float32, shape=[batch_size * 3, 512, 300, c], name='input_holder')
output_holder = tf.placeholder(tf.int32, shape=[batch_size * 3, ], name='output_holder')  # not use, but you can use
logits, feature_speaker = net(in_put=input_holder)  # not used,but you can use

margin_loss, compute_acc = triplet_loss(feature_speaker)  #
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(margin_loss)
# correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), output_holder)
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:

    if load_model:
        saver.restore(sess,
                      # tf.train.latest_checkpoint(checkpoint_dir=saved_model_path))
                      tf.train.latest_checkpoint(checkpoint_dir=saved_model_path))
    else:
        sess.run(tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('logs', sess.graph_def)

    worker_summary_dict = {'op': summary_op, 'writer': summary_writer}
    summary_op = worker_summary_dict['op']
    summary_writer = worker_summary_dict['writer']

    train_data, train_label = load_data(
        saved_data_path='data_set/data_set.npy',
        saved_label_path='data_set/label_set.npy')

    stepsssss = 0
    for epoch in range(n_epoch):
        start_time = time.time()
        # training
        train_loss = 0.0  # not used
        train_acc = 0.0   # not used
        early_stop_loss = 0.1  # early stop train
        init_loss = 9e10  # we must set very big init loss

        triplet_index = mini_batch_triplet(train_label)
        keys_data = triplet_index.keys()
        while init_loss > early_stop_loss:
            a_p_n = list(sample(keys_data, batch_size * 2))  # 选出类别
            pair_data_a_p = a_p_n[:batch_size]  # 正例索引  0--8, 1--7, 2--9 ,3---6 ,4===9
            signal_data_n = a_p_n[batch_size:batch_size * 2]  # 反例索引
            # 正例训练样本
            all_loss_for_batch = []
            mini_triplet_batch = []
            for cur_a_p, cur_n in zip(pair_data_a_p, signal_data_n):
                index_a_p = list(sample(triplet_index[cur_a_p], 2))
                index_n = list(sample(triplet_index[cur_n], 1))
                mini_triplet_batch.append(train_data[index_a_p[0]])
                mini_triplet_batch.append(train_data[index_a_p[-1]])
                mini_triplet_batch.append(train_data[index_n[-1]])  # (96, 512, 300, 3)
            _, a_p_n_user_loss, acc_ = sess.run([train_op,  # (96, 1024)
                                                 margin_loss,
                                                 compute_acc],
                                                feed_dict={input_holder: np.array(mini_triplet_batch)})
            stepsssss += 1
            # train_loss += np.sum(a_p_n_user_loss)
            # train_acc += acc
            init_loss = a_p_n_user_loss
            train_acc = acc_
            print("   train loss: ", init_loss / batch_size)
            print("   train acc:", train_acc)
            summary = tf.Summary()
            summary.value.add(tag='episode_loss',
                              simple_value=float(init_loss / batch_size))
            summary.value.add(tag='episode_acc',
                              simple_value=float(train_acc))
            summary_writer.add_summary(summary, stepsssss)
            summary_writer.flush()
            if epoch % 2 == 0:
                saver.save(sess, model_path, global_step=epoch)

