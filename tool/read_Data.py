# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as  tf
import numpy as np
import cv2


def bbox_flip_left_right(bboxes, w):
    w = tf.to_float(w)
    y1, x1, y2, x2, cls = tf.split(bboxes, 5, axis=1)
    x1, x2 = w - 1. - x2, w - 1. - x1
    return tf.concat([y1, x1, y2, x2, cls], axis=1)


def parse(se):
    features = tf.parse_single_example(
        se, features={
            'im': tf.FixedLenFeature([], tf.string),
            'label': tf.VarLenFeature(tf.int64),

        }
    )
    img = features['im']
    label = features['label']
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.decode_raw(img, tf.uint8)
    # img = tf.reshape(img, (32, -1, 3))

    img = tf.image.rgb_to_grayscale(img)

    inds = label.indices[:, 0]
    value = label.values
    l = tf.to_int32(tf.to_float(tf.shape(img)[1]) / 4)
    inds = inds[:l]
    value = value[:l]
    return img, inds, value, l


def readData(files, batch_size=1, num_epochs=20000, num_threads=16, shuffle_buffer=1024, num_shards=1, shard_index=0):
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(parse, num_parallel_calls=num_threads)

    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None, 1], [None], [None], []),
                                   padding_values=(np.array(127, dtype=np.uint8), np.array(0, dtype=np.int64),
                                                   np.array(-1, dtype=np.int64), np.array(0, dtype=np.int32)))

    # dataset = dataset.batch(batch_size)

    iter = dataset.make_initializable_iterator()
    return iter


def draw_gt(im, gt):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    for box in boxes:
        # print(box)
        y1, x1, y2, x2 = box[:4]
        print(box)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
        print(box[-1])
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


def creat_label(inds, value):
    m = tf.shape(value)[0]
    n = tf.shape(value)[1]
    m = tf.to_int64(m)
    n = tf.to_int64(n)
    b = tf.ones_like(inds, dtype=tf.int32) * (tf.range(tf.shape(inds)[0])[..., None])
    b = tf.to_int64(b)
    b = tf.reshape(b, (-1, 1))
    inds = tf.reshape(inds, (-1, 1))
    value = tf.reshape(value, (-1,))
    inds = tf.concat([b, inds], axis=-1)
    index = value >= 0
    inds = tf.boolean_mask(inds, index)
    value = tf.boolean_mask(value, index)
    label = tf.SparseTensor(indices=inds, values=value, dense_shape=[m, n])
    return label


def decode(v):
    v = [chars_dic[x] for x in v]
    return ''.join(v)


if __name__ == "__main__":
    import os
    from datetime import datetime

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from sklearn.externals import joblib

    a = [chr(ord('0') + i) for i in range(10)]
    b = [chr(ord('a') + i) for i in range(26)]
    c = [chr(ord('A') + i) for i in range(26)]
    chars = a + b + c
    chars_dic = {}
    for i in range(len(chars)):
        chars_dic[i] = chars[i]
        chars_dic[chars[i]] = i
    print(chars_dic)
    path = '/home/zhai/PycharmProjects/Demo35/CRNN/data_process/'
    files = [path + 'mnt.tf']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    Iter = readData(files, batch_size=3, shuffle_buffer=None)

    im, inds, value, length = Iter.get_next()
    label = creat_label(inds, value)

    with tf.Session() as sess:
        sess.run(Iter.initializer)
        for i in range(10):
            img, inds_, value_, b_, length_ = sess.run([im, inds, value, label, length])

            for j in range(img.shape[0]):
                v = value_[j]
                v = v[v >= 0]
                cv2.imshow('a', img[j])
                print(decode(v), img.shape)
                cv2.waitKey(2000)
            cv2.destroyAllWindows()

    # label = creat_label(inds, value)
    # net_out = tf.placeholder(dtype=tf.float32, shape=(11, 1, 63))
    # label = tf.to_int32(label)
    # loss = tf.reduce_mean(
    #     tf.nn.ctc_loss(labels=label, inputs=net_out, sequence_length=length))
    # with tf.Session(config=config) as sess:
    #
    #     sess.run(Iter.initializer)
    #     for i in range(1):
    #         t = np.random.random((11, 1, 63))
    #         t = t.astype(np.float32)
    #         loss_, label_, length_ = sess.run([loss, label, length], feed_dict={net_out: t})
    #         print(np.array(label_))
    #         print(length_)
    #
    #         pass
    # for i in range(10000000000):
    #
    #     img, inds_, value_, b_,length_ = sess.run([im, inds, value, label,length])
    #     print(i)
    #     print('********************')
    # print(img.shape)
    # print(inds_.shape)
    #
    # print(value_.shape)
    # print(length_)
    # # print(img.shape,  )
    # # print(inds_)
    # # print(value_)
    # # cv2.imshow('a', img)
    # # # print(label_[i])
    # # cv2.waitKey(2000)
    # # cv2.destroyAllWindows()
    # print(img.max(),img.min())
    # if i>1:
    #     for j in range(img.shape[0]):
    #         v=value_[j]
    #         v=v[v>=0]
    #         cv2.imshow('a', img[j])
    #         print(decode(v),length_)
    #         cv2.waitKey(2000)
    #     cv2.destroyAllWindows()

    # tt = mask_[0]
    #     # print(tt.shape, num_[0], bbox[0].shape)

    #     #
    #
    #     # print(datetime.now(),i,mask.shape)
    #     # print(e,f,g,h,h[0]==False)
    #     print(datetime.now(), i)
    # #
    # draw_gt(a,b)
    # joblib.dump((a,b),'(t_im,t_bboxes).pkl')
