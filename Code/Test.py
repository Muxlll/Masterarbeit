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

random.seed(0)
seed(0)
tf.random.set_seed(0)


def relu_advanced(x):
    return K.relu(x)


ACTIVATION_FUNCTION = relu_advanced

# INITIALIZER = tf.keras.initializers.RandomNormal(stddev=0.05, seed=42)

# INITIALIZER = tf.keras.initializers.Constant(value=0.1)

# INITIALIZER = tf.keras.initializers.GlorotUniform(seed=42)


def create_model(learning_rate=0.0001, input_dim=10, number_layers=1, neurons_per_layer=20):
    # create model
    model = Sequential()
    # model.add(Dense(neurons_per_layer, input_shape=(input_dim,), activation=ACTIVATION_FUNCTION,
    #                 kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
    # for _ in range(number_layers):
    #     model.add(Dense(neurons_per_layer, activation=ACTIVATION_FUNCTION,
    #                     kernel_initializer=INITIALIZER, bias_initializer=INITIALIZER))
    # model.add(Dense(1, activation=None))

    model.add(Dense(neurons_per_layer, input_shape=(input_dim,), activation=ACTIVATION_FUNCTION))
    for _ in range(number_layers):
        model.add(Dense(neurons_per_layer, activation=ACTIVATION_FUNCTION))
    model.add(Dense(1, activation=None))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


dataset = HPO.Dataset(task_id=233211)


def reset_seeds():
    np.random.seed(1)
    random.seed(2)
    tf.random.set_seed(3)


def evaluate_model(epochs, batch_size, learning_rate, number_of_layers, neurons_per_layer):

    # return epochs + batch_size + learning_rate + number_of_layers + neurons_per_layer

    kfold = KFold(n_splits=3)

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


for _ in range(2):
    print(evaluate_model(40, 100, 1e-04, 1, 20))
