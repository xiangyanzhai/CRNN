# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
from tool.read_Data import readData
from tool.config import Config
import os

is_training = True
_BATCH_DECAY = 0.999
from datetime import datetime


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
        grads = tf.concat(grads, 0)
        grad = tf.reduce_mean(grads, 0)
        grad_and_var = (grad, grad_and_vars[0][1])
        # [(grad0, var0),(grad1, var1),...]
        average_grads.append(grad_and_var)
    return average_grads


class CRNN():
    def __init__(self, config):
        self.config = config
        self.__hidden_num = 256
        self.__layers_num = 2
        self.__num_classes = config.num_cls + 1

    def create_label(self, inds, values):
        m = tf.shape(values)[0]
        n = tf.shape(values)[1]
        m = tf.to_int64(m)
        n = tf.to_int64(n)
        b = tf.ones_like(inds, dtype=tf.int32) * (tf.range(tf.shape(inds)[0])[..., None])
        b = tf.to_int64(b)
        b = tf.reshape(b, (-1, 1))
        inds = tf.reshape(inds, (-1, 1))
        value = tf.reshape(values, (-1,))
        inds = tf.concat([b, inds], axis=-1)
        index = value >= 0
        inds = tf.boolean_mask(inds, index)
        value = tf.boolean_mask(value, index)
        label = tf.SparseTensor(indices=inds, values=value, dense_shape=[m, n])
        label = tf.to_int32(label)
        return label

    def __feature_sequence_extraction(self, input_tensor):
        is_training = self.config.is_train
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None):
            net = slim.repeat(input_tensor, 2, slim.conv2d, 64, kernel_size=3, stride=1, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=3, stride=1, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, kernel_size=3, stride=1, scope='conv3')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool3')
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv4')
            net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn4')
            net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv5')
            net = slim.batch_norm(net, decay=_BATCH_DECAY, is_training=is_training, scope='bn5')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
            net = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv6')
        return net

    def __map_to_sequence(self, input_tensor):
        shape = input_tensor.get_shape().as_list()

        assert shape[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(input_tensor, axis=1)

    def __sequence_label(self, input_tensor, input_sequence_length):
        with tf.variable_scope('LSTM_Layers'):
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num] * self.__layers_num]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num] * self.__layers_num]
            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, input_tensor, sequence_length=input_sequence_length, dtype=tf.float32)

            [batch_size, _, hidden_num] = input_tensor.get_shape().as_list()
            hidden_num = self.__hidden_num * 2
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_num])

            # Doing the affine projection
            w = tf.Variable(tf.truncated_normal([hidden_num, self.__num_classes], stddev=0.01), name="w")
            logits = tf.matmul(rnn_reshaped, w)

            logits = tf.reshape(logits, [tf.shape(input_tensor)[0], -1, self.__num_classes])
            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
        return rnn_out, raw_pred

    def build_network(self, images, sequence_length=None):
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(images)

        # cnn_out=tf.placeholder(dtype=tf.float32,shape=(1,1,25,512))
        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(input_tensor=cnn_out)
        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(input_tensor=sequence, input_sequence_length=sequence_length)

        return net_out

    def build_net(self, Iter):
        img, inds, values, length = Iter.get_next()
        label = self.create_label(inds, values)
        img = tf.to_float(img)
        img.set_shape(tf.TensorShape([None, 32, None, 1]))
        with tf.variable_scope('CRNN'):
            net_out = self.build_network(img, length)

        loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=label, inputs=net_out, sequence_length=length))
        return loss

    def train(self):
        shard_nums = self.config.gpus
        batch_size = self.config.gpus * self.config.batch_size_per_GPU
        print('******************* batch_size', batch_size)
        print('*******************  shard_nums', shard_nums)
        base_lr = self.config.lr
        print('******************* base_lr', base_lr)
        steps = tf.Variable(0.0, name='crnn', trainable=False)
        x = 2
        lr = tf.case({steps < 120000.0 * x: lambda: base_lr,
                      steps < 200000.0 * x: lambda: base_lr / 10},
                     default=lambda: base_lr / 100)

        tower_grads = []
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        # opt=tf.train.AdadeltaOptimizer(lr,0.9)
        var_reuse = False
        Iter_list = []

        c = 0

        for i in range(shard_nums):
            with tf.device('/gpu:%d' % i):
                print('************ shard_index', c)
                Iter = readData(self.config.files, batch_size=self.config.batch_size_per_GPU,
                                num_threads=16,
                                shuffle_buffer=1024,
                                num_shards=shard_nums, shard_index=c)
                Iter_list.append(Iter)
                with tf.variable_scope('', reuse=var_reuse):
                    if c == 0:
                        loss = self.build_net(Iter)

                    else:
                        loss = self.build_net(Iter)

                    var_reuse = True

                c += 1

                train_vars = tf.trainable_variables()
                l2_loss = tf.losses.get_regularization_losses()
                l2_re_loss = tf.add_n(l2_loss)
                crnn_loss = loss + l2_re_loss
                grads_and_vars = opt.compute_gradients(crnn_loss, train_vars)
                tower_grads.append(grads_and_vars)

        for v in tf.global_variables():
            print(v)

        grads = average_gradients(tower_grads)
        grads = list(zip(*grads))[0]

        grads, norm = tf.clip_by_global_norm(grads, 10.0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(zip(grads, train_vars), global_step=steps)
        saver = tf.train.Saver(max_to_keep=200)
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        file='/home/zhai/PycharmProjects/Demo35/CRNN/train/models/CRNN.ckpt-120000'

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            saver.restore(sess, file)

            for Iter in Iter_list:
                sess.run(Iter.initializer)

            for i in range(120000, int(250200 * x)):
                if i % 20 == 0:

                    _, crnn_loss_, loss_, l2_re_loss_, norm_, lr_ = sess.run(
                        [train_op, crnn_loss, loss, l2_re_loss, norm, lr])

                    print(datetime.now(), 'crnn_loss:%.4f' % crnn_loss_, 'loss:%.4f' % loss_,
                          'l2_re_loss:%.4f' % l2_re_loss_, 'norm:%.4f' % norm_, i, lr_)

                else:
                    sess.run(train_op)

                if (i + 1) % 5000 == 0 or ((i + 1) % 1000 == 0 and i < 10000) or (i + 1) == int(250000 * x):
                    saver.save(sess, os.path.join('./models/', 'CRNN_2x.ckpt'), global_step=i + 1)
            saver.save(sess, os.path.join('./models/', 'CRNN_2x.kpt'), global_step=i + 1)


if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    path = '/home/zhai/PycharmProjects/Demo35/CRNN/data_process/'
    files = [path + 'mnt.tf']
    config = Config(True, Mean, files, num_cls=62, lr=0.01, batch_size_per_GPU=64, gpus=1)

    crnn = CRNN(config)

    crnn.train()
