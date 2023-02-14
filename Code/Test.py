import HPO

import pysgpp

import sys

import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

import sklearn.metrics

X = []
Y = []

num_samples = 100

for i in range(1, num_samples):
    X.append(2.0*math.pi/num_samples * float(i))
    Y.append(math.sin(2.0*math.pi/num_samples * float(i)))

#plt.plot(X, Y)
#plt.show()

X = torch.Tensor(X)
Y = torch.Tensor(Y)

X = X.reshape(-1, 1)

dataset = HPO.Dataset(X, Y)

hyperparameterspace = {
    'learning_rate' : [0.00001, 0.01],
    'epochs': [1, 60]
}

def to_standard(lower, upper, value):
    return (value-lower)/(upper-lower)


def from_standard(lower, upper, value):
    return value*(upper-lower)+lower

class ExampleFunction(pysgpp.ScalarFunction):
    """Example objective function from the title of my Master's thesis."""

    def __init__(self):
        super(ExampleFunction, self).__init__(2)


    def eval(self, x):
        
        # Function to create model, required for KerasClassifier
        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(1, input_shape=(1,), activation='relu'))
            model.add(Dense(1, activation=None))
            # Compile model
            opt = keras.optimizers.Adam(learning_rate=x[0])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        model = KerasRegressor(model=create_model, verbose=2)

        model.fit(X, Y, epochs=int(x[1]))
        Y_predicted = model.predict(X)

        return sklearn.metrics.mean_squared_error(Y.tolist(), Y_predicted.tolist())
################################## generate Grid ##################################

pysgpp.omp_set_num_threads(1)

pysgpp.Printer.getInstance().setVerbosity(5)

f = ExampleFunction()
# dimension of domain
d = f.getNumberOfParameters()
# B-spline degree
p = 4
# maximal number of grid points
N = 100
# adaptivity of grid generation
gamma = 0.9


grid = pysgpp.Grid.createModBsplineGrid(d, p)
gridGen = pysgpp.OptIterativeGridGeneratorRitterNovak(f, grid, N, gamma)

functionValues = gridGen.getFunctionValues()

if not gridGen.generate():
    print("Grid generation failed, exiting.")
    sys.exit(-1)

gridStorage = grid.getStorage()

x_values = []
y_values = []
for i in range(gridStorage.getSize()):
    gp = gridStorage.getPoint(i)
    x_values.append(gp.getStandardCoordinate(0)) 
    y_values.append(gp.getStandardCoordinate(1))
    
    
plt.plot(x_values, y_values, 'bo')

######################################## grid functions ########################################
# Hierarchization
functionValues = gridGen.getFunctionValues()
coeffs = pysgpp.DataVector(len(functionValues))
hierSLE = pysgpp.HierarchisationSLE(grid)
sleSolver = pysgpp.AutoSLESolver()

if not sleSolver.solve(hierSLE, gridGen.getFunctionValues(), coeffs):
    print("Solving failed, exiting.")
    sys.exit(1)

# define interpolant and gradient
ft = pysgpp.InterpolantScalarFunction(grid, coeffs)
ftGradient = pysgpp.InterpolantScalarFunctionGradient(grid, coeffs)
gradientDescent = pysgpp.OptGradientDescent(ft, ftGradient)
x0 = pysgpp.DataVector(d)

##################### find point with minimal loss (which are already evaluated) #################

# find point with smallest value as start point for gradient descent
x0Index = 0
fX0 = functionValues[0]
for i in range(1, len(functionValues)):
    if functionValues[i] < fX0:
        fX0 = functionValues[i]
        x0Index = i

x0 = gridStorage.getCoordinates(gridStorage.getPoint(x0Index));
ftX0 = ft.eval(x0)

print("\nOptimal hyperparameters so far:")
print("Epochs: ", from_standard(1,300,x0[1]))
print("learning_rate: ", from_standard(0.00001,0.01,x0[0]))

print("Resulting loss:")
print(ftX0)

################################## Optimize with gradient descent ##################################
#print("x0 = {}".format(x0))
#print("f(x0) = {:.6g}, ft(x0) = {:.6g}\n".format(fX0, ftX0))

## We apply the gradient method and print the results.
gradientDescent.setStartingPoint(x0)
gradientDescent.optimize()
xOpt = gradientDescent.getOptimalPoint()
ftXOpt = gradientDescent.getOptimalValue()

print(xOpt)
fXOpt = f.eval(xOpt)

print("\nOptimal hyperparameters after optimization:")
print("Epochs: ", from_standard(1,300,xOpt[1]))
print("learning_rate: ", from_standard(0.00001,0.01,xOpt[0]))
print("Resulting loss (Optimal value from optimization):")
print(ftXOpt)
print("Resulting loss (Optimal point evaluated):")
print(fXOpt)
#print("\nxOpt = {}".format(xOpt))
#print("f(xOpt) = {:.6g}, ft(xOpt) = {:.6g}\n".format(fXOpt, ftXOpt))