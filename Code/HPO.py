import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import timeit


from sklearn.utils import shuffle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import itertools

from mpl_toolkits import mplot3d


from scipy.io import arff
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import time, sys
from IPython.display import clear_output

import pysgpp

from bayes_opt import BayesianOptimization, UtilityFunction


def update_progress(progress, time, remaining_time):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    text += "\nCurrent time per iteration: " + str(time)
    text += "\nApprox. time remaining: " + str(remaining_time)
    print(text)

    
def to_standard(lower, upper, value):
    return (value-lower)/(upper-lower)


def from_standard(lower, upper, value):
    return value*(upper-lower)+lower


class Dataset:
    def __init__(self, X, Y, ratio=0.8) -> None:
        self.X = X
        self.Y = Y
        X, Y = shuffle(X, Y)

        self.X_train = torch.Tensor(X[:int(len(X) * ratio)])
        self.X_test = torch.Tensor(X[int(len(X) * ratio):])
        self.Y_train = torch.Tensor(Y[:int(len(Y) * ratio)])
        self.Y_test = torch.Tensor(Y[int(len(Y) * ratio):])

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_Y_train(self):
        return self.Y_train

    def get_Y_test(self):
        return self.Y_test

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    

"""
    Optimization class
    arguments: Dataset, model, hyperparameter space, type of optimization, metric
        Type:
            0: Grid search
            1: Random search
            2: Bayesian Optimization
            3: Sparse grid search
        Metrics:
            "accuracy": simple testing accuracy
            "loss": simple loss
            "10-fold crossvalidation": 10-fold crossvalidation       
"""
class Optimization:
    def __init__(self, dataset, model, hyperparameterspace, type=0) -> None:
        self.dataset = dataset   
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.type = type    


    def fit(self):
        if self.type == 0:
            clf = GridSearchCV(self.model, self.hyperparameterspace)
            print(self.dataset.get_X())
            return clf.fit(self.dataset.get_X(), self.dataset.get_Y())
        elif self.type == 1:
            return ((0,0), 1)
        elif self.type == 2:
            return ((0,0), 1)
        elif self.type == 3:
            return ((0,0), 1)
        else:
            AssertionError("Type not specified correctly")