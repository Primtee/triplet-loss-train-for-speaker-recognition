# coding=utf-8

import time
import numpy as np   # 多维数据处理模块
import tensorflow as tf
# from dataprovider.model.vgg_model import Net
from create.data_management import load_data, mini_batch
__author__ = 'NXG'

# 数据集地址
path = 'D:/flower/flower_photos/'
data_set_path = 'data_set/'
# 模型保存地址
model_path = 'model/model_flower'
load_model = True
saved_model_path = 'model/'
# 将所有的图片resize成100*100
w = 512
h = 300
c = 3
n_epoch = 250

def net(in_put, n_class=100, train=True, if_regularizer=True):
        """
        VGG网络结构
        :param train: 是否是训练
        :param if_regularizer: 是否正则
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
                tf.add_to_collection('losses', regularizer_method(conv6_weights))  # 正则化矩阵
            conv6_biases = tf.get_variable("bias", [4096], initializer=tf.constant_initializer(0.0))
            conv6 = tf.nn.conv2d(pool3, conv6_weights, strides=[1, 1, 1, 1], padding='SAME', name='conv6')
            relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases), 'relu6')
            if train:
                relu6 = tf.nn.dropout(relu6, 0.6)  # 过拟合
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
            ful_1 = tf.nn.dropout(ful_1, 0.5)  # 过拟合
            print('ful_1:', ful_1)
        with tf.name_scope("layer13-fullconnection_1"):
            ful_2 = tf.contrib.layers.fully_connected(ful_1, num_outputs=n_class)
            ful_2 = tf.nn.relu(ful_2, name='ful_2')
            print('ful_2:', ful_2)
        return ful_2


input_holder = tf.placeholder(tf.float32, shape=[None, 512, 300, c], name='input_holder')
output_holder = tf.placeholder(tf.int32, shape=[None, ], name='output_holder')
logits = net(in_put=input_holder)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=output_holder)
# 设置整体学习率为α为0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# 设置预测精度
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), output_holder)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:

    if load_model:
        saver.restore(sess,
                      tf.train.latest_checkpoint(checkpoint_dir=saved_model_path))
    else:
        sess.run(tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('logs', sess.graph_def)

    worker_summary_dict = {'op': summary_op, 'writer': summary_writer}
    summary_op = worker_summary_dict['op']
    summary_writer = worker_summary_dict['writer']
    # train_data, train_label = load_data(saved_data_path='F:/zns/speaker_voice_system/dataprovider/data_set/data_set.npy',
    #                                     saved_label_path='F:/zns/speaker_voice_system/dataprovider/data_set/label_set.npy')
    # test_data, test_label = load_data(saved_data_path='F:/zns/speaker_voice_system/dataprovider/data_set/test_data_set.npy',
    #                                   saved_label_path='F:/zns/speaker_voice_system/dataprovider/data_set/test_abel_set.npy')

    train_data, train_label = load_data(
        saved_data_path='data_set/data_set.npy',
        saved_label_path='data_set/label_set.npy')

    stepsssss = 0
    for epoch in range(n_epoch):
        start_time = time.time()
        # training#训练集
        train_loss = 0.0
        train_acc = 0.0
        batch_size = 32
        # for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        train_batch_data, train_batch_label = mini_batch(cur_data=train_data,
                                                         cur_label=train_label,
                                                         shuffle_data=True,
                                                         batch_size=32)
        for index, (train_batch, label_batch) in enumerate(zip(train_batch_data, train_batch_label)):
            stepsssss += 1
            # print('shape of train data:', np.array(train_batch).shape)
            _, err, ac = sess.run([train_op, loss, acc],
                                  feed_dict={input_holder: np.array(train_batch),
                                             output_holder: label_batch})
            # print('err:', err)
            # print('ac:', ac)
            train_loss += np.sum(err)
            train_acc += ac
            print("   train loss: ", (train_loss / (index + 1)))
            print("   train acc:" , (train_acc / (index + 1)))
            summary = tf.Summary()
            summary.value.add(tag='episode_loss',
                              simple_value=float(train_loss / (index + 1)))
            summary.value.add(tag='episode_acc',
                              simple_value=float(train_acc / (index + 1)))
            summary_writer.add_summary(summary, stepsssss)
            summary_writer.flush()
        # del train_data
        # del train_batch_data
        # train_data1, train_label1 = load_data(
        #     saved_data_path='data_set/data_set600.npy',
        #     saved_label_path='data_set/label_set600.npy')
        # train_batch_data1, train_batch_label1 = mini_batch(cur_data=train_data1,
        #                                                    cur_label=train_label1,
        #                                                    shuffle_data=True,
        #                                                    batch_size=32)
        # # batch two
        # for index1, (train_batch1, label_batch1) in enumerate(zip(train_batch_data1, train_batch_label1)):
        #     stepsssss += 1
        #     _, err1, ac1 = sess.run([train_op, loss, acc],
        #                             feed_dict={input_holder: np.array(train_batch1),
        #                                        output_holder: label_batch1})
        #     train_loss += np.sum(err1)
        #     train_acc += ac1
        #     print("   train loss:" , float(train_loss / (index + index1 + 1 + 1)))
        #     print("   train acc:" , float(train_acc / (index + index1 + 1 + 1)))
        #     summary = tf.Summary()
        #     summary.value.add(tag='episode_loss',
        #                       simple_value=float(train_loss / (index + index1 + 1 + 1)))
        #     summary.value.add(tag='episode_acc',
        #                       simple_value=float(train_acc / (index + index1 + 1 + 1)))
        #     summary_writer.add_summary(summary, stepsssss)
        #     summary_writer.flush()
        # del train_data1
        # del train_batch_data1
        if epoch % 2 == 0:
             saver.save(sess, model_path, global_step=epoch)

