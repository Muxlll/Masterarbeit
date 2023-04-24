import openml

from openml import tasks

import HPO

import pysgpp

import sys

import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sklearn.metrics


from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K

from sklearn.model_selection import KFold, cross_val_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from scikeras.wrappers import KerasRegressor, KerasClassifier

hyperparameterspace = {
    'loss': ["list", 'mean_absolute_error', 'mean_squared_error'],
    'epochs': ["interval-int", 1, 20],
    #'batch_size': ["interval-int", 40, 160],
    #'optimizer__learning_rate': ["interval", 0.0000001, 0.0001]
}

hyperparameterspace_special = {}
for key in hyperparameterspace.keys():
    liste = []
    for i in range(1, len(hyperparameterspace[key])):
        liste.append(hyperparameterspace[key][i])
    hyperparameterspace_special[key] = liste

dataset = HPO.Dataset([],[])

class ExampleFunction(pysgpp.ScalarFunction):

    def __init__(self):
        super(ExampleFunction, self).__init__(1)


    def eval(self, x):
        return 1#x[0] + x[1]

f = ExampleFunction()

ies = []
budgets = []

for i in range(100):
    BUDGET = i
    optimization = HPO.SparseGridSearchOptimization(dataset, f, hyperparameterspace, budget=BUDGET, verbosity=0, degree=2, adaptivity=0.95, optimizer="gradient_descent")

    result = optimization.fit()
    budgets.append(result[2])
    ies.append(i)

    
print(ies)
print(budgets)