#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from chainer.training import extensions
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as f:
        dic = pickle.load(f, encoding='latin-1')
    return dic


def make_train_data():
    x_train = None
    y_train = []
    for num in range(1, 6):
        data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(num))
        if x_train is None:
            x_train = data_dic['data']
        else:
            x_train = np.vstack((x_train, data_dic['data']))
        y_train += data_dic['labels']
    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    y_train = np.array(y_train)
    x_train = x_train.astype(np.float32)
    x_train /= 255
    y_train = y_train.astype(np.int32)
    return x_train, y_train


def make_test_data():
    test_data = unpickle("cifar-10-batches-py/test_batch")
    x_test = test_data['data']
    x_test = x_test.reshape(len(x_test), 3, 32, 32)
    y_test = np.array(test_data['labels'])
    x_test = x_test.astype(np.float32)
    x_test /= 255
    y_test = y_test.astype(np.int32)
    return x_test, y_test


def make_model():
    model = chainer.FunctionSet(conv1=F.Convolution2D(3, 32, 2, pad=1),
        conv2=F.Convolution2D(32, 32, 2, pad=1),
        conv3=F.Convolution2D(32, 32, 2, pad=1),
        conv4=F.Convolution2D(32, 32, 2, pad=1),
        conv5=F.Convolution2D(32, 32, 2, pad=1),
        conv6=F.Convolution2D(32, 32, 2, pad=1),
        l1=F.Linear(512, 512),
        l2=F.Linear(512, 10))
    return model


def forward(model, x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.relu(model.conv1(x))
    h = F.max_pooling_2d(F.relu(model.conv2(h)), 2, stride=2)
    h = F.relu(model.conv3(h))
    h = F.max_pooling_2d(F.relu(model.conv4(h)), 2, stride=2)
    h = F.relu(model.conv5(h))
    h = F.max_pooling_2d(F.relu(model.conv6(h)), 2, stride=2)
    h = F.dropout(F.relu(model.l1(h)), train=train)
    y = model.l2(h)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


def main():
    x_train, y_train = make_train_data()
    x_test, y_test = make_test_data()
    print(len(x_train))
    model = make_model()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    N = 50000
    N_test = 10000
    batch_size = 100
    for epoch in range(40):
        print("epoch", epoch+1)

        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0

        for i in range(0, N, batch_size):
            x_batch = np.asarray(x_train[perm[i:i+batch_size]])
            y_batch = np.asarray(y_train[perm[i:i+batch_size]])

            optimizer.zero_grads()
            loss, acc = forward(model, x_batch, y_batch)
            loss.backward()
            optimizer.update()

            train_loss.append(loss.data)
            train_acc.append(acc.data)
            sum_loss += float(loss.data) * batch_size
            sum_accuracy += float(acc.data) * batch_size

        print("train mean loss={}, accuracy={}".format(sum_loss/N, sum_accuracy/N))

        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, N_test, batch_size):
            x_batch = np.asarray(x_train[perm[i:i+batch_size]])
            y_batch = np.asarray(y_train[perm[i:i+batch_size]])

            loss, acc = forward(model, x_batch, y_batch)

            test_loss.append(loss.data)
            test_acc.append(acc.data)
            sum_loss += float(loss.data) * batch_size
            sum_accuracy += float(acc.data) * batch_size

        print("test mean loss={}, accuracy={}".format(sum_loss/N_test, sum_accuracy/N_test))


if __name__ == '__main__':
    main()
