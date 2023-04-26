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
    'epochs': ["interval-int", 1, 20],
    'batch_size': ["interval-int", 10, 200],
    'optimizer__learning_rate': ["interval-log", 0.0000001, 0.1],
    'model__neurons_per_layer': ["interval-int", 1, 100],
    'model__number_of_layers': ["interval-int", 1, 10],
}

hyperparameterspace_special = {}
for key in hyperparameterspace.keys():
    liste = []
    for i in range(1, len(hyperparameterspace[key])):
        liste.append(hyperparameterspace[key][i])
    hyperparameterspace_special[key] = liste


# ids = [233211]
# task = tasks.get_task(ids[0])
# dataset = task.get_dataset()

# print("Current dataset:", i, "of", len(ids), "with name:", dataset.name)

# # Get the data itself as a dataframe (or otherwise)
# data, target, categorical_indicator, names = dataset.get_data(dataset.default_target_attribute, dataset_format="array")

# # split into categorical and numerical features
# categorical_features = [[x[i] for i in range(len(x)) if categorical_indicator[i]] for x in data]
# numerical_features = [[x[i] for i in range(len(x)) if not categorical_indicator[i]] for x in data]

# # one hot encoding of the categorical one
# encoder = OneHotEncoder(sparse_output=False).fit(categorical_features)
# transformed = encoder.transform(categorical_features)

# # additional scaling of numerical features
# scaler = StandardScaler().fit(numerical_features)
# numerical_features = scaler.transform(numerical_features)

# # bring back together
# data = [numerical_features[i].tolist() + transformed[i].tolist() for i in range(len(numerical_features))]#

# ## additional scaling 
# #scaler = StandardScaler().fit(data)
# #data = scaler.transform(data)

# scaler = StandardScaler().fit(target.reshape(-1,1))
# target = scaler.transform(target.reshape(-1,1))

# X = torch.Tensor(data)
# Y = torch.Tensor(target)

ids = [233211, 359935, 359952, 359931, 359949, 359938]
#[359940, 317614, 359934, 359946, 359932, 233214, 359943]

for id in ids:
    dataset = HPO.Dataset(ratio=0.9,task_id=id)

    def relu_advanced(x):
        return K.relu(x)

    ACTIVATION_FUNCTION = relu_advanced
    INITIALIZER = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1,seed=42)


    def evaluate_model(loss, epochs, batch_size, model_learning_rate, neurons_per_layer, number_of_layers):
        # Function to create model, required for KerasClassifier
        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(neurons_per_layer, input_shape=(len(dataset.get_X()[0]),), activation=ACTIVATION_FUNCTION, kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
            for _ in range(number_of_layers):
                model.add(Dense(neurons_per_layer, activation=ACTIVATION_FUNCTION, kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
            model.add(Dense(1, activation=None))
            # Compile model
            
            optimizer = keras.optimizers.Adam(learning_rate=model_learning_rate)

            model.compile(loss=loss, optimizer=optimizer)
            return model


        kfold = KFold(n_splits=4)

        X = dataset.get_X_train().tolist() + dataset.get_X_validation().tolist()
        Y = dataset.get_Y_train().tolist() + dataset.get_Y_validation().tolist()
        
    
        X += dataset.get_X_test().tolist()
        Y += dataset.get_Y_test().tolist()

        X = np.array(X)
        Y = np.array(Y)

        split = (kfold.split(X, Y))

        values = []

        for i, (train_index, test_index) in enumerate(split):
            X_train = X[train_index]
            Y_train = Y[train_index]

            X_val = X[test_index]
            Y_val = Y[test_index]


            model = KerasRegressor(model=create_model, verbose=0)

            model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

            Y_predicted = model.predict(X_val)
            error = sklearn.metrics.mean_squared_error(Y_predicted, Y_val)
            values.append(error)

            K.clear_session()
            del model

        result = sum(values)/len(values)
        return result


    loss = 'mean_squared_error'#hyperparameterspace_special["loss"][index]

    epochs = 10

    batch_size = 57

    model_learning_rate = 7.873896236216029e-02

    neurons_per_layer = 40 # int(params[3])

    number_of_layers = 1 # int(params[4])

    for _ in range(1):
        print(evaluate_model(loss, epochs, batch_size, model_learning_rate, neurons_per_layer, number_of_layers))

    print("")

    