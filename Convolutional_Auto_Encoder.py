# -*- coding: utf-8 -*-

import sys, os, cv2 ,glob
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, optimizers, Chain, datasets
from chainer import training
from itertools import chain
import cm
import cupy as cp



# convolutinal_Autoencoderの定義
class CAE(Chain):

    def __init__(self):
        super(CAE, self).__init__()
        with self.init_scope():
            self.encoder1 = L.ConvolutionND(ndim=2, in_channels=1, out_channels=18,ksize=7, stride=1)
            self.encoder2 = L.ConvolutionND(ndim=2, in_channels=18, out_channels=45, ksize=5, stride=1, pad=2)
            self.decoder1 = L.DeconvolutionND(ndim=2, in_channels=45, out_channels=64,
                                              ksize=1, stride=1)# , outsize=(61, 41))
            self.decoder2 = L.DeconvolutionND(ndim=2, in_channels=64, out_channels=1,
                                              ksize=7, stride=1)#, outsize=(128, 88))
            self.Linear1 = L.Linear(None, out_size=2048)
            self.Linear2 = L.Linear(in_size=2048, out_size=1024)
            self.Linear3 = L.Linear(in_size=1024, out_size=2048)
            self.decoder3 = L.DeconvolutionND(ndim=2, in_channels=16, out_channels=45,
                                              ksize=(15, 13), stride=1, outsize=(30, 20))  # , outsize=(128, 88))
            # self.decoder1 = L.DeconvolutionND(ndim=2, in_channels=45, out_channels=64,
            #                                   ksize=(32, 22), stride=1)  # , outsize=(61, 41))
            # self.decoder2 = L.DeconvolutionND(ndim=2, in_channels=64, out_channels=32,
            #                                   ksize=(62, 42), stride=1)  # , outsize=(128, 88))
            # self.decoder3 = L.DeconvolutionND(ndim=2, in_channels=32, out_channels=1, ksize=7, stride=1)


    def __call__(self, x):

        # encoder-part
        h = F.relu(self.encoder1(x))
        print(h.shape)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        print(h.shape)
        h = F.relu(self.encoder2(h))
        print(h.shape)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        print(h.shape)
        h = F.relu(self.Linear1(h))
        print(h.shape)
        h = F.relu(self.Linear2(h))
        print(h.shape)


        # decoder-part
        h = F.relu(self.Linear3(h))
        print(h.shape)
        h = F.reshape(h, (h.shape[0], 16, 16, 8))
        print("reshape:" + str(h.shape))

        h = F.relu(self.decoder3(h))
        print(h.shape)
        h = F.unpooling_2d(h, ksize=3, stride=2, outsize=(61, 41))
        print(h.shape)
        h = F.relu(self.decoder1(h))
        print(h.shape)
        h = F.unpooling_2d(h, ksize=2, stride=2, outsize=(122, 82))
        print(h.shape)
        h = self.decoder2(h)

        print(h.shape)
        return h

        '''
        h = F.relu(self.encoder1(x))
        print(h.shape)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        print(h.shape)
        h = F.relu(self.encoder2(h))
        print(h.shape)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        print(h.shape)
        h = F.unpooling_2d(h, ksize=3, stride=2, outsize=(61, 41))
        print(h.shape)
        h = F.relu(self.decoder1(h))
        print(h.shape)
        h = F.unpooling_2d(h, ksize=2, stride=2, outsize=(122, 82))
        print(h.shape)
        h = self.decoder2(h)

        print(h.shape)
        return h
        '''

        '''
        h = F.relu(self.encoder1(x))
        print(h.shape)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        print(h.shape)
        h = F.relu(self.encoder2(h))
        print(h.shape)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        print(h.shape)
        h = F.relu(self.decoder1(h))
        print(h.shape)
        h = F.unpooling_2d(h, ksize=1, stride=1, outsize=(61, 41))
        print(h.shape)
        h = self.decoder2(h)
        print(h.shape)
        # h = F.unpooling_2d(h, ksize=2, stride=2, outsize=(128, 88))
        h = self.decoder3(h)

        print(h.shape)
        return h
        
        '''
