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

from sklearn.linear_model import SGDRegressor


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

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

INITIALIZER = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=42)


def create_model(learning_rate=0.001, input_dim=10):
    # create model
    model = Sequential()
    model.add(Dense(30, input_shape=(input_dim,), activation=ACTIVATION_FUNCTION,
                kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
    model.add(Dense(30, activation=ACTIVATION_FUNCTION,
                kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
    model.add(Dense(1, activation=None))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    return model


numeric_features = [not x for x in dataset.get_categorical_indicator()]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")),
           ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        #("selector", SelectPercentile(chi2, percentile=50)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, dataset.get_categorical_indicator()),
    ]
)

# onehotencoder = ColumnTransformer(
#     transformers=[
#         ("categorical", OneHotEncoder(sparse_output=False),
#             dataset.get_categorical_indicator())
#     ], remainder='passthrough'
# )

# final regressor
regressor = TransformedTargetRegressor(regressor=KerasRegressor(model=create_model, input_dim=dataset.get_input_dim(), verbose=0),
                                       transformer=StandardScaler())

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

pipeline.fit(dataset.get_X_train(),
             dataset.get_Y_train(), regressor__epochs=15)

print(pipeline['regressor'].regressor_.history_['loss'])

# summarize history for loss
plt.plot(pipeline['regressor'].regressor_.history_['loss'])
plt.title('model training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

test_pre = pipeline.predict(dataset.get_X_test())
print(dataset.get_Y_test())
print(test_pre)
print(sklearn.metrics.mean_absolute_error(test_pre, dataset.get_Y_test()))

val_pre = pipeline.predict(dataset.get_X_validation())

print(sklearn.metrics.mean_absolute_error(val_pre, dataset.get_Y_validation()))


