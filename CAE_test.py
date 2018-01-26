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
sys.path.append("/home/common-ns/PycharmProjects/Prepare")
from tools import load_OULP
sys.path.append("/home/common-ns/PycharmProjects/models")
from Convolutional_Auto_Encoder import CAE


def plot_mnist_data(samples, cnt):
    if cnt == 0:
        for index, (data, label) in enumerate(samples):
            plt.subplot(2, batch_size, index + 1)
            plt.axis('off')
            plt.imshow(data.reshape(128, 88), interpolation='nearest', cmap='gray')

            n = int(label)
            plt.title(n, color='blue')
            cnt = cnt + 1

    if cnt == 1:
        for index, (data, label) in enumerate(samples):
            plt.subplot(2, batch_size, index + 11)
            plt.axis('off')
            plt.imshow(data.reshape(128, 88), interpolation='nearest', cmap='gray')

            n = int(label)
            plt.title(n, color='red')
        plt.show()
    list = np.array(samples)
    np.save("/home/common-ns/setoguchi/chainer_files/CAE.npy", list)


if __name__ == '__main__':
    print "a"
    model = L.Classifier(CAE(), lossfun=F.sigmoid_cross_entropy)
    serializers.load_npz("/home/common-ns/setoguchi/chainer_files/Convolutional_Auto_Encoder/result_SCE/model_snapshot_146000", model)
    test_dir = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/CV02/*"
    test = load_OULP(test_dir)
    # print(len(test))
    test = test[0:10]  # 0から10取り出す
    batch_size = 10
    # print (len(test))

    print "a"

    count = 0
    source = []
    # 原画像の表示
    for (data, label) in test:
        source.append((data, label))
    plot_mnist_data(source, count)
    count = count + 1
    print "a"
    pred_list = []
    # 復元画像の表示
    for (data, label) in test:
        pred_data = model.predictor(np.array([data]).astype(np.float32)).data
        # pred_data_cpu = cuda.to_cpu(pred_data)
        item = model.__getitem__("loss")
        print(pred_data.shape)
        # print(item)
        pred_list.append((pred_data, label))
    plot_mnist_data(pred_list, count)

