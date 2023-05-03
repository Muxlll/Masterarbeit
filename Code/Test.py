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
from sklearn.model_selection import GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor


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
    'regressor__regressor__epochs': ["interval-int", 1, 5],
    'regressor__regressor__batch_size': ["interval-int", 10, 200],
    'regressor__regressor__model__optimizer__learning_rate': ["interval-log", 0.0000001, 0.1]
}

hyperparameterspace_special = {}
for key in hyperparameterspace.keys():
    liste = []
    for i in range(1, len(hyperparameterspace[key])):
        liste.append(hyperparameterspace[key][i])
    hyperparameterspace_special[key] = liste


dataset = HPO.Dataset(task_id=233211)


def relu_advanced(x):
    return K.relu(x)


ACTIVATION_FUNCTION = relu_advanced
INITIALIZER = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=42)

def evaluate_model(loss, epochs, batch_size, model_learning_rate, neurons_per_layer, number_of_layers):
    # Function to create model, required for KerasClassifier
    def create_model(learning_rate=0.0001, input_dim=10):
        # create model
        model = Sequential()
        model.add(Dense(30, input_shape=(input_dim,), activation=ACTIVATION_FUNCTION, kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
        model.add(Dense(30, activation=ACTIVATION_FUNCTION, kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
        model.add(Dense(1, activation=None))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model


    kfold = KFold(n_splits=2)

    split = (kfold.split(dataset.get_X(), dataset.get_Y()))

    values = []

    # partial one hot encoding
    onehotencoder = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(sparse_output=False),
            dataset.get_categorical_indicator())
        ], remainder='passthrough'
    )

    # final regressor
    regressor = TransformedTargetRegressor(regressor=KerasRegressor(model=create_model, input_dim=dataset.get_input_dim(), verbose=0),
                                    transformer=StandardScaler())


    pipeline = Pipeline([
        ('ohencoder', onehotencoder),
        ('standardizer', StandardScaler(with_mean=False)),
        ('regressor', regressor)
    ])

    for i, (train_index, test_index) in enumerate(split):
        X_train = dataset.get_X()[train_index]
        Y_train = dataset.get_Y()[train_index]

        X_val = dataset.get_X()[test_index]
        Y_val = dataset.get_Y()[test_index]


        model = KerasRegressor(model=create_model, verbose=0)

        pipeline.fit(X_train, Y_train, regressor__epochs=epochs, regressor__batch_size=batch_size)

        Y_predicted = pipeline.predict(X_val)
        error = sklearn.metrics.mean_absolute_error(Y_predicted, Y_val)
        values.append(error)

        K.clear_session()
        del model

    result = sum(values)/len(values)
    return result

def blackboxfunction(params):
    #index = int(params[0]*(len(hyperparameterspace_special["loss"])-1))
    loss = 'mean_squared_error'#hyperparameterspace_special["loss"][index]
    
    epochs = int(params[0])

    batch_size = int(params[1])

    model_learning_rate = params[2]

    neurons_per_layer = 40 # int(params[3])

    number_of_layers = 1 # int(params[4])

    return evaluate_model(loss, epochs, batch_size, model_learning_rate, neurons_per_layer, number_of_layers)
        

class ExampleFunction(pysgpp.ScalarFunction):

    def __init__(self):
        super(ExampleFunction, self).__init__(len(hyperparameterspace.keys()))


    def eval(self, x):
        #index = int(x[0]*(len(hyperparameterspace_special["loss"])-1))
        loss = 'mean_squared_error'#hyperparameterspace_special["loss"][index]
        
        epochs = int(HPO.from_standard(hyperparameterspace_special["epochs"][0], hyperparameterspace_special["epochs"][1], x[0]))

        batch_size = int(HPO.from_standard(hyperparameterspace_special["batch_size"][0], hyperparameterspace_special["batch_size"][1], x[1]))

        model_learning_rate = HPO.from_standard_log(hyperparameterspace_special["optimizer__learning_rate"][0], hyperparameterspace_special["optimizer__learning_rate"][1], x[2])
        
        neurons_per_layer = 40 # int(HPO.from_standard(hyperparameterspace_special["model__neurons_per_layer"][0], hyperparameterspace_special["model__neurons_per_layer"][1], x[3]))

        number_of_layers = 1 # int(HPO.from_standard(hyperparameterspace_special["model__number_of_layers"][0], hyperparameterspace_special["model__number_of_layers"][1], x[4]))

        return evaluate_model(loss, epochs, batch_size, model_learning_rate, neurons_per_layer, number_of_layers)

opt = HPO.SparseGridSearchOptimization(dataset=dataset, model=ExampleFunction(), hyperparameterspace=hyperparameterspace, budget=3)

result = opt.fit()

print(result[0].best_params_)
print(-result[0].best_score_)
