# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
from tool.config import Config
import os
import codecs

is_training = False
_BATCH_DECAY = 0.99
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
            print(batch_size, hidden_num)
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

    def process_im(self, im):
        H = tf.shape(im)[0]
        W = tf.shape(im)[1]
        H = tf.to_float(H)
        W = tf.to_float(W)
        scale = 32. / H
        nh = H * scale
        nw = W * scale
        nh = tf.to_int32(nh)
        nw = tf.to_int32(nw)

        im = tf.image.resize_images(im, (32, nw))
        im = tf.cast(im, tf.uint8)
        return im

    def build_net(self):
        self.input = tf.placeholder(dtype=tf.string)
        img = tf.image.decode_jpeg(self.input)
        img = self.process_im(img)
        img=img[...,::-1]
        img = tf.image.rgb_to_grayscale(img)
        print(img)
        img = tf.to_float(img)

        img = img[None]

        img.set_shape(tf.TensorShape([None, 32, None, 1]))
        with tf.variable_scope('CRNN'):
            net_out = self.build_network(img)

        ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, [tf.shape(net_out)[0]], merge_repeated=False)

        # ctc_decoded, ct_log_prob = tf.nn.ctc_greedy_decoder(net_out, [tf.shape(net_out)[0]], merge_repeated=True)
        print('***********', ctc_decoded)
        self.de = ctc_decoded

    def decode(selfs, v):
        if len(v) == 0:
            return ''
        v = [chars_dic[x] for x in v]
        return ''.join(v)

    def test(self):
        self.build_net()

        pre_file = r'D:\PycharmProjects\Demo36\CRNN\train\models\CRNN.ckpt-250000'

        saver = tf.train.Saver()
        path = r'D:\dataset\mjsynth/mnt/ramdisk/max/90kDICT32px/'
        file_test = r'D:\dataset\mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_test.txt'
        file = file_test
        print(file)
        with codecs.open(file, 'r', 'utf-8') as f:
            S = f.read().strip()
            S = S.split('\n')
        print(len(S))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            saver.restore(sess, pre_file)

            i = 0
            m = len(S)
            m = 1000
            c = 0
            # np.random.seed(50)
            # np.random.shuffle(S)
            for s in S[:m]:

                i += 1
                try:
                    s = s.strip().split()[0]
                    label = s.split('_')[1]
                    img_file = path + s
                    img = tf.gfile.FastGFile(img_file, 'rb').read()

                    de = sess.run(self.de, feed_dict={self.input: img})

                    v = de[0].values
                    pre = self.decode(v)
                    # print(pre,label,pre==label)
                    if (pre == label):
                        c += 1
                    # else:
                    #     print(datetime.now(),i,pre,label)
                except:
                    continue
                if (i) % 1000 == 0:
                    print(datetime.now(), c, i, c / i)

        print(c)
        print(c / m)


a = [chr(ord('0') + i) for i in range(10)]
b = [chr(ord('a') + i) for i in range(26)]
c = [chr(ord('A') + i) for i in range(26)]
chars = a + b + c
chars_dic = {}
for i in range(len(chars)):
    chars_dic[i] = chars[i]
    chars_dic[chars[i]] = i
if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    path = '/home/zhai/PycharmProjects/Demo35/CRNN/'
    # files = [path + 'voc_07.tf', path + 'voc_12.tf']
    files = [path + 'mnt.tf']
    # files=[path+'coco_train2017_scale.tf']
    config = Config(True, Mean, files, num_cls=62, lr=0.01, batch_size_per_GPU=64, gpus=1)

    crnn = CRNN(config)

    crnn.test()
# 1x
# 0.8914260920456495
# 2x
# 0.9035560085074227
