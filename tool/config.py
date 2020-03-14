# !/usr/bin/python
# -*- coding:utf-8 -*-
class Config():
    def __init__(self, is_train, Mean, files, lr=1e-3, weight_decay=0.0005,
                 num_cls=20,
                 batch_size_per_GPU=1, gpus=1,
                 bias_lr_factor=1

                 ):
        self.is_train = is_train
        self.Mean = Mean
        self.files = files
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_cls = num_cls
        self.batch_size_per_GPU = batch_size_per_GPU
        self.gpus = gpus

        self.bias_lr_factor=bias_lr_factor
        print('CRNN')
        print('==============================================================')
        print('Mean:\t', self.Mean)
        print('files:\t', self.files)
        print('lr:\t', self.lr)
        print('weight_decay:\t', self.weight_decay)
        print('num_cls:\t', self.num_cls)
        print('batch_size_per_GPU:\t', self.batch_size_per_GPU)
        print('gpus:\t', self.gpus)


        print('==============================================================')
        print('bias_lr_factor:\t',self.bias_lr_factor)
        print('==============================================================')
