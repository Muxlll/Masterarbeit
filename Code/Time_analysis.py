import HPO
import pysgpp
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.metrics
from sklearn.model_selection import KFold
import numpy as np
import keras
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from numpy.random import seed
import random
import time
import os
import math
import timeit

random.seed(1)
seed(2)
tf.random.set_seed(3)

def reset_seeds():
    np.random.seed(1)
    random.seed(2)
    tf.random.set_seed(3)

VERBOSE = 1
CV = 2 #[(slice(None), slice(None))]
TESTING = True 

DATASETS = []

GRID_RESULT = []
RANDOM_RESULT = []
BAYESIAN_RESULT = []
SPARSE_RESULT = []
SPARSE_RESULT_OPTIMIZED = []

GRID_COST = []
RANDOM_COST = []
BAYESIAN_COST = []
SPARSE_COST = []
SPARSE_COST_OPTIMIZED = []


hyperparameterspace = {
    'epochs': ["interval-int", 1, 20],
    'batch_size': ["interval-int", 1, 200],
    'learning_rate': ["interval-log", 0.000000001, 0.1],
    #'number_layers': ["interval-int", 1, 20],
    #'neurons_per_layer': ["interval-int", 1, 50]
}

hyperparameterspace_special = {}
for key in hyperparameterspace.keys():
    liste = []
    for i in range(1, len(hyperparameterspace[key])):
        liste.append(hyperparameterspace[key][i])
    hyperparameterspace_special[key] = liste


def create_model(learning_rate=0.0001, input_dim=10, number_layers=1, neurons_per_layer=20):
    # create model
    model = Sequential()

    model.add(Dense(neurons_per_layer, input_shape=(input_dim,), activation='relu'))
    for _ in range(number_layers):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(1, activation=None))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


dataset = HPO.Dataset(task_id=233211)

def evaluate_model(epochs, batch_size, learning_rate, number_of_layers, neurons_per_layer, deterministic=True):

    # # ################# RASTRIGIN #################
    # sum = 0
    # sum += epochs ** 2 - 10 * math.cos(2 * math.pi * epochs)
    # sum += batch_size ** 2 - 10 * math.cos(2 * math.pi * batch_size)
    # sum += learning_rate ** 2 - 10 * math.cos(2 * math.pi * learning_rate)
    # sum += number_of_layers ** 2 - 10 * math.cos(2 * math.pi * number_of_layers)
    # sum += neurons_per_layer ** 2 - 10 * math.cos(2 * math.pi * neurons_per_layer)
    # return len(hyperparameterspace) * 10 + sum


    ################# EGGHOLDER #################
    # return -epochs * math.sin(math.sqrt(abs(epochs - (batch_size + 47)))) - (batch_size + 47) * math.sin(math.sqrt(abs((batch_size + 47 + 0.5 * epochs))))

    # return epochs + batch_size + learning_rate + number_of_layers + neurons_per_layer

    kfold = KFold(n_splits=CV)

    split = (kfold.split(dataset.get_X(), dataset.get_Y()))

    values = []

    numeric_features = [not x for x in dataset.get_categorical_indicator()]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(
                handle_unknown="infrequent_if_exist", sparse_output=False)),
            # ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer,
                dataset.get_categorical_indicator()),
        ]
    )

    for i, (train_index, test_index) in enumerate(split):

        if deterministic:
            reset_seeds()

        X_train = dataset.get_X()[train_index]
        Y_train = dataset.get_Y()[train_index]

        X_val = dataset.get_X()[test_index]
        Y_val = dataset.get_Y()[test_index]

        preprocessor.fit(X_train, Y_train)

        X_train = preprocessor.transform(X_train)
        X_val = preprocessor.transform(X_val)

        regressor = KerasRegressor(model=create_model,
                                    learning_rate=learning_rate,
                                    input_dim=len(
                                        X_train[0]),
                                    number_layers=number_of_layers,
                                    neurons_per_layer=neurons_per_layer,
                                    verbose=0)

        regressor = TransformedTargetRegressor(regressor=regressor,
                                                transformer=StandardScaler())

        regressor.fit(X_train, Y_train, epochs=epochs,
                        batch_size=batch_size, shuffle=False)

        Y_predicted = regressor.predict(X_val)
        # error = sklearn.metrics.mean_absolute_error(Y_predicted, Y_val)
        error = sklearn.metrics.mean_absolute_percentage_error(
            Y_predicted, Y_val)
        values.append(error)

        del regressor
        K.clear_session()

    result = sum(values)/len(values)
    return result


starttime = timeit.default_timer()

evaluate_model(40, 10, 0.0001, 1, 30)

endtime = timeit.default_timer()

print(endtime - starttime)

