#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Support Vector Machine
scikit-learnに入っている数値の画像データサンプルを利用して、SVMを試してみる
'''

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    digits = load_digits(2)
    data_train, data_test, label_train, label_test = train_test_split(digits['data'], digits['target'])
    estimator = LinearSVC(C=1.0)
    estimator.fit(data_train, label_train)
    label_predict = estimator.predict(data_test)
    print(confusion_matrix(label_test, label_predict))
    print(accuracy_score(label_test, label_predict))


if __name__ == '__main__':
    main()
