# -*- coding: utf-8 -*-

import sys, os, cv2 ,glob
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, optimizers, Chain, datasets, serializers
from chainer.training import extensions, StandardUpdater, Trainer
from itertools import chain
import cm
import cupy as cp
import sys
sys.path.append("/home/common-ns/PycharmProjects/Prepare")
from tools import load_OULP
sys.path.append("/home/common-ns/PycharmProjects/models")
from Convolutional_Auto_Encoder import CAE
sys.path.append("/home/common-ns/PycharmProjects/Test")
import CAE_test



def train():
    # train_txt = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV01.txt"
    train_dir = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/CV01(Gallery)/*"
    train = load_OULP(path_dir=train_dir)
    
    # print(train[0])
    
    # test_txt = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV02.txt"
    test_dir = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Probe/CV02(Probe)/*"
    test = load_OULP(path_dir=test_dir)
    
    # 教師データ
    # train = train[0:1000]
    train = [i[0] for i in train]
    train = datasets.TupleDataset(train, train)
    batch_size = 239
    train_iter = chainer.iterators.SerialIterator(train, batch_size=batch_size)
    
    # テスト用データ
    test = test[0:239]
    
    #model = L.Classifier(Autoencoder(), lossfun=F.mean_squared_error)
    model = L.Classifier(CAE(), lossfun=F.mean_squared_error)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    
    updater = StandardUpdater(train_iter, optimizer, device=-1)
    trainer = Trainer(updater, (1, 'epoch'), out="result", )
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    # trainer.extend(extensions.snapshot(), trigger=(250, 'epoch'))
    trainer.extend(extensions.snapshot_object(target=model, filename='model_snapshot_{.updater.iteration}'), trigger=(250, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
    serializers.save_npz("/home/common-ns/setoguchi/chainer_files/Convolutional_Auto_Encoder/CAE_model", model)
    
if __name__ == '__main__':
    train()
    # CAE_test()