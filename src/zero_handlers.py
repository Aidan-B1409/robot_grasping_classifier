# Robot Grip Classifier
# AI-539 Machine Learning Challenges @ OSU
# Aidan Beery
# 03-06-2022


import numpy as np


def DummyZeroHandler(X):
    return X


def DropEffort(X):
    return X.drop(X.filter(like='eff').columns, axis=1)


def IncrementFeatureSpace(X):
    return np.add(X, 1)
