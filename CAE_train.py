# -*- coding: utf-8 -*-

import sys, os, cv2 ,glob
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, optimizers, Chain, datasets, serializers, functions
from chainer.training import extensions, StandardUpdater, Trainer
from itertools import chain
import cm
import cupy as cp

from chainer.utils import type_check

import sys
sys.path.append("/home/common-ns/PycharmProjects/Prepare")
from tools import load_OULP
sys.path.append("/home/common-ns/PycharmProjects/models")
from Convolutional_Auto_Encoder import CAE
sys.path.append("/home/common-ns/PycharmProjects/Test")
import CAE_test


# chainerのsigmoid_cross_entropyはtarget値はint32しか受けつけないので、変更する
class SCELoss(functions.SigmoidCrossEntropy):
    ignore_label = -1

    def __init__(self, normalize=True, reduce='mean'):
        super(SCELoss, self).__init__()
        self.normalize = normalize
        if reduce not in ('mean', 'no'):
            raise ValueError(
                "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
                'given' % reduce)
        self.reduce = reduce
        self.count = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            t_type.dtype.kind == 'f',  # targetにfloat32も受けるようにする ここ重要
            x_type.shape == t_type.shape
        )

def sce_loss(x, t, normalize=True, reduce='mean'):
    return SCELoss(normalize, reduce).apply((x, t))[0]



def train():
    # train_txt = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/CV01.txt"
    train_dir = "/media/common-ns/New Volume/reseach/Dataset/OU-ISIR_by_Setoguchi/Gallery/CV01(Gallery)/*"
    train = load_OULP(path_dir=train_dir)
    
    # print(train[0])
    
    # 教師データ
    # train = train[0:1000]
    train = [i[0] for i in train] # dataのパスとラベルのうち、dataだけ抜き出す
    train = datasets.TupleDataset(train, train)   # 同じパス画像のペアから、dataに変換してタプルにする

    batch_size = 195
    train_iter = chainer.iterators.SerialIterator(train, batch_size=batch_size)

    
    #model = L.Classifier(Autoencoder(), lossfun=F.mean_squared_error)
    model = L.Classifier(CAE(), lossfun=sce_loss)
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    
    updater = StandardUpdater(train_iter, optimizer, device=0)
    trainer = Trainer(updater, (1000, 'epoch'), out="result", )
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.extend(extensions.snapshot(), trigger=(200, 'epoch'))
    trainer.extend(extensions.snapshot_object(target=model, filename='model_snapshot_{.updater.iteration}'), trigger=(250, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
    serializers.save_npz("/home/common-ns/setoguchi/chainer_files/Convolutional_Auto_Encoder/CAE_FC_model", model)
    
if __name__ == '__main__':
    train()
    # CAE_test()