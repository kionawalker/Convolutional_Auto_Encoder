# _*_ coding:utf-8 _*_

import sys, os, cv2 ,glob
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, optimizers, Chain, datasets, serializers
from chainer import training
from itertools import chain
import sys
sys.path.append("~/PycharmProject/Prepare")
import tools
sys.path.append("~/PycharmProject/models")
from Convolutional_Auto_Encoder import CAE


def plot_mnist_data(samples):
    for index, (data, label) in enumerate(samples):
        plt.subplot(2, batch_size, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(128, 88), interpolation='nearest')

        n = int(label)
        plt.title(n, color='blue')

    for index, (data, label) in enumerate(samples):
        plt.subplot(2, batch_size, index + 10)
        plt.axis('off')
        plt.imshow(data.reshape(128, 88), interpolation='nearest')

        n = int(label)
        plt.title(n, color='red')
    list = np.array(samples)
    np.save("/home/common-ns/setoguchi/chainer_files/CAE.npy", list)


if __name__=='__main__':

    model = model = L.Classifier(CAE(), lossfun=F.sigmoid_cross_entropy)
    serializers.load_npz("/home/common-ns/setoguchi/chainer_files/"
                                 "Convolutional_Auto_Encoder/CAE_model", model)
    test_dir = "/media/common-ns/New Volume/reseach/Dataset" \
               "/OU-ISIR_by_Setoguchi/Probe/CV02(Probe)/"
    test = tools.load_OULP(test_dir)
    test = test[0:10]  # 0から10取り出す
    batch_size = 10

    source = []
    # 原画像の表示
    for (data, label) in test:
        source.append((data, label))
    plot_mnist_data(source)

    pred_list = []
    # 復元画像の表示
    for (data, label) in test:
        pred_data = model.predictor(np.array([data]).astype(np.float32)).data
        # pred_data_cpu = cuda.to_cpu(pred_data)
        item = model.__getitem__("loss")
        print(pred_data.shape)
        print(item)
        pred_list.append((pred_data, label))
    plot_mnist_data(pred_list)