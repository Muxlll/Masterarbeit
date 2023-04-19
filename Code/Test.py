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

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import numpy as np
import keras
from keras import backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from scikeras.wrappers import KerasRegressor, KerasClassifier

hyperparameterspace = {
    'loss': ["list", 'mean_absolute_error', 'mean_squared_error'],
    'epochs': ["interval-int", 1, 20],
    'batch_size': ["interval-int", 40, 160],
    'optimizer__learning_rate': ["interval", 0.0000001, 0.0001]
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

dataset = HPO.Dataset(task_id=233214)

print(len(dataset.get_X_train()))
print(len(dataset.get_X_validation()))
print(len(dataset.get_X_test()))

def relu_advanced(x):
    return K.relu(x)

ACTIVATION_FUNCTION = relu_advanced

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_shape=(len(dataset.get_X()[0]),), activation=ACTIVATION_FUNCTION))
    model.add(Dense(30, activation=ACTIVATION_FUNCTION))
    model.add(Dense(1, activation=None))
    # Compile model
    
    optimizer = keras.optimizers.Adam(learning_rate=0.004937375642917827)

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

model = KerasRegressor(model=create_model, verbose=0)

history = model.fit(dataset.get_X_train(), dataset.get_Y_train(), epochs=6)

plt.plot(history.history_['loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


Y_predicted_train = model.predict(dataset.get_X_train())

print("Training error:", sklearn.metrics.mean_squared_error(dataset.get_Y_train(), Y_predicted_train))

Y_predicted_test = model.predict(dataset.get_X_test())

print("Testing error:", sklearn.metrics.mean_squared_error(dataset.get_Y_test(), Y_predicted_test))
