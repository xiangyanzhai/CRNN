# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import codecs
import cv2
import tensorflow as tf
from PIL import Image
from datetime import datetime

file_train = '/home/zhai/PycharmProjects/Demo35/dataset/mnt/ramdisk/max/90kDICT32px/annotation_train.txt'
file_test = '/home/zhai/PycharmProjects/Demo35/dataset/mnt/ramdisk/max/90kDICT32px/annotation_test.txt'
file_val = '/home/zhai/PycharmProjects/Demo35/dataset/mnt/ramdisk/max/90kDICT32px/annotation_val.txt'
file = '/home/zhai/PycharmProjects/Demo35/dataset/mnt/ramdisk/max/90kDICT32px/annotation.txt'
imlist = '/home/zhai/PycharmProjects/Demo35/dataset/mnt/ramdisk/max/90kDICT32px/imlist.txt'
lexicon = '/home/zhai/PycharmProjects/Demo35/dataset/mnt/ramdisk/max/90kDICT32px/lexicon.txt'


def _float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def getexample(im, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'im': _bytes(im),
        'label': _int64(label),
    }))


def process_data(path, file, tf_name):
    print(file)
    with codecs.open(file, 'r', 'utf-8') as f:
        S = f.read().strip()
        S = S.split('\n')
    print(len(S))
    writer = tf.python_io.TFRecordWriter(tf_name)
    i = 0
    e = 0
    # for s in S[4029:4030]:
    np.random.seed(50)
    np.random.shuffle(S)
    for s in S[:]:
        # print(i)
        i += 1
        if i%10000==0:
            print(i)
        s = s.strip().split()[0]
        label = s.split('_')[1]

        img_file = path + s
        try:
            im = Image.open(img_file)
        except:
            print('++++++++++++++++++++')
            e += 1
            continue
        im = np.array(im)
        if not (len(im.shape) == 3 and im.shape[-1] == 3):
            print('-------------------')
            e += 1
            continue
        img = cv2.imread(img_file)

        if img is None:
            continue
        h, w = img.shape[:2]
        nw = int(w * 32 / h)
        if int(nw / 4) <= len(label):
            print('aaaaaaaaaa')
            continue
        im = cv2.resize(im, (nw, 32))
        Label = []
        for l in label:
            Label.append(chars_dic[l])
        Label = np.array(Label, dtype=np.int64)
        im=cv2.imencode('.jpg',im)[1].tobytes()
        example = getexample(im, Label)

        writer.write(example.SerializeToString())
    writer.close()
    print('***********', e)
    pass


a = [chr(ord('0') + i) for i in range(10)]
b = [chr(ord('a') + i) for i in range(26)]
c = [chr(ord('A') + i) for i in range(26)]
chars = a + b + c
chars_dic = {}
for i in range(len(chars)):
    chars_dic[i] = chars[i]
    chars_dic[chars[i]] = i
print(chars_dic)

if __name__ == "__main__":
    path = '/home/zhai/PycharmProjects/Demo35/dataset/mnt/ramdisk/max/90kDICT32px/'
    file = file_train
    tf_name = 'mnt.tf'
    process_data(path, file, tf_name)

    pass
