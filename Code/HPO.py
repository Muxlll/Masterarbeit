import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import timeit

import copy

import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize


from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt

import time
import sys
from IPython.display import clear_output

import pysgpp


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
    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100)
    text += "\nCurrent time per iteration: " + str(time)
    text += "\nApprox. time remaining: " + str(remaining_time)
    print(text)


class Dataset:
    def __init__(self, X, Y, ratio=0.8) -> None:
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        X, Y = shuffle(X, Y)

        self.X_train = torch.Tensor(X[:int(len(X) * ratio)])
        self.X_validation = torch.Tensor(self.X_train[int(len(self.X_train) * ratio):])
        self.X_train = torch.Tensor(self.X_train[:int(len(self.X_train) * ratio)])
        self.X_test = torch.Tensor(X[int(len(X) * ratio):])
        self.Y_train = torch.Tensor(Y[:int(len(Y) * ratio)])
        self.Y_validation = torch.Tensor(self.Y_train[int(len(self.Y_train) * ratio):])
        self.Y_train = torch.Tensor(self.Y_train[:int(len(self.Y_train) * ratio)])
        self.Y_test = torch.Tensor(Y[int(len(Y) * ratio):])

    def get_X_train(self):
        return self.X_train

    def get_X_validation(self):
        return self.X_validation
    
    def get_X_test(self):
        return self.X_test

    def get_Y_train(self):
        return self.Y_train
    
    def get_Y_validation(self):
        return self.Y_validation

    def get_Y_test(self):
        return self.Y_test

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * \
            (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]
    
    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    time = 0

    for n in range(n_iters):

        starttime = timeit.default_timer()

        model.fit(xp, yp)

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(
                bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model,
                                           yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(
                expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(
                bounds[:, 0], bounds[:, 1], bounds.shape[0])

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

        percentage = (n+1)/n_iters

        endtime = timeit.default_timer()
        time += (endtime-starttime)

        remaining_time_prediction = (time/(n+1))*n_iters - time

        update_progress(percentage, (endtime-starttime),
                        remaining_time_prediction)
        
    print("Iterations took", time, "seconds")

    return xp, yp


def to_standard(lower, upper, value):
    return (value-lower)/(upper-lower)


def from_standard(lower, upper, value):
    return value*(upper-lower)+lower


"""
    Optimization class 
    Params:
        dataset:                data of the class Dataset
        model:                  model to find the best params for (can be model or blackbox function depending on type)
        hyperparameterspace:    definition of hyperparameter space (dict with "list" or "interval" as first element of list)
        type:                   available ones: "grid_search", "random_search", "bayesian", "sparse"
        cv:                     k parameter for cross validation (k-fold cv)
        scoring:                metric for evaluating model
        budget:                 upper bound for number of model evaluations
        verbosity:              verbosity for output
        sparse_params:          list of values: [B-spline_degree, adaptivity, optimizer] 
                                        optimizers: one of ["adaptive_gradient_descent", "adaptive_newton", "bfgs", "cmaes", 
                                                            "differential_evolution", "gradient_descent", "nlcg", "nelder_mead", 
                                                            "newton", "rprop"]
"""
class OldOptimization:
    def __init__(self, 
                 dataset: Dataset, 
                 model, 
                 hyperparameterspace: dict, 
                 type: str = "grid_search", 
                 cv: int = 5, 
                 scoring: str = 'neg_mean_squared_error', 
                 budget: int = 100, 
                 verbosity: int = 1, 
                 sparse_params: list = [3, 0.95, "gradient_descent"]) -> None:
        

        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.type = type
        self.cv = cv
        self.scoring = scoring
        self.budget = budget
        self.verbosity = verbosity
        self.sparse_params = sparse_params

        if self.type == "grid_search" or self.type == "random_search":

            param_dimension = len(self.hyperparameterspace)
            param_per_dimension = int(self.budget**(1/param_dimension))

            for key in self.hyperparameterspace.keys():
                if self.hyperparameterspace.get(key)[0] == "list":
                    self.hyperparameterspace_processed.get(key).pop(0)
                    self.hyperparameterspace_processed[key] = self.hyperparameterspace_processed.get(key)
                elif self.hyperparameterspace.get(key)[0] == "interval":
                    param_list = []
                    for i in range(param_per_dimension):
                        upper = self.hyperparameterspace.get(key)[2]
                        lower = self.hyperparameterspace.get(key)[1]
                        param_list.append(
                            (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2)
                    self.hyperparameterspace_processed[key] = param_list
                elif self.hyperparameterspace.get(key)[0] == "interval-int":
                    param_list = []
                    for i in range(param_per_dimension):
                        upper = self.hyperparameterspace.get(key)[2]
                        lower = self.hyperparameterspace.get(key)[1]
                        param_list.append(int(
                            (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2))
                    self.hyperparameterspace_processed[key] = param_list
                else:
                    print("Need to specify the type of list")
        elif self.type == "bayesian":
            list = []
            for key in self.hyperparameterspace.keys():
                if self.hyperparameterspace.get(key)[0] != 'list':
                    self.hyperparameterspace_processed.get(key).pop(0)
                    list.append(self.hyperparameterspace_processed.get(key))
                else: 
                    list.append([0,1])
            self.hyperparameterspace_processed = np.array(list)
        elif self.type == "sparse":
            for key in self.hyperparameterspace.keys():
                self.hyperparameterspace_processed.get(key).pop(0)

    def fit(self):
        if self.type == "grid_search":
            clf = GridSearchCV(
                self.model, self.hyperparameterspace_processed, cv=self.cv, scoring=self.scoring, error_score='raise', verbose=self.verbosity)
            X_fit = torch.cat((self.dataset.get_X_train(), self.dataset.get_X_validation()))
            Y_fit = torch.cat((self.dataset.get_Y_train(), self.dataset.get_Y_validation()))
            return clf.fit(X_fit, Y_fit)

        elif self.type == "random_search":

            clf = RandomizedSearchCV(
                self.model, self.hyperparameterspace_processed, cv=self.cv, scoring=self.scoring, verbose=self.verbosity, n_iter=self.budget)
            X_fit = torch.cat((self.dataset.get_X_train(), self.dataset.get_X_validation()))
            Y_fit = torch.cat((self.dataset.get_Y_train(), self.dataset.get_Y_validation()))
            return clf.fit(X_fit, Y_fit)

        elif self.type == "bayesian":
            return bayesian_optimisation(self.budget, self.model, self.hyperparameterspace_processed)
        
        elif self.type == "sparse":

            f = self.model

            # dimension of domain
            d = f.getNumberOfParameters()
            # B-spline degree
            p = self.sparse_params[0]
            # maximal number of grid points
            N = self.budget
            # adaptivity of grid generation
            gamma = self.sparse_params[1]
            # choice of optimizer
            optimizer_choice = self.sparse_params[2]

            grid = pysgpp.Grid.createModBsplineGrid(d, p)
            gridGen = pysgpp.OptIterativeGridGeneratorRitterNovak(
                f, grid, N, gamma)

            functionValues = gridGen.getFunctionValues()
            if not gridGen.generate():
                print("Grid generation failed, exiting.")
                sys.exit(-1)

            
            gridStorage = grid.getStorage()
            if d == 2:
                x_values = []
                y_values = []
                z_values = []
                for i in range(gridStorage.getSize()):
                    gp = gridStorage.getPoint(i)
                    x_values.append(gp.getStandardCoordinate(0))
                    y_values.append(gp.getStandardCoordinate(1))
                    z_values.append(functionValues[i])

                #if self.verbosity >= 1:
                #    plt.plot(x_values, y_values, 'bo')

                if self.verbosity >= 1:
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')

                    ax.plot_trisurf(x_values, y_values, z_values)
                    plt.show()

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
            x0 = pysgpp.DataVector(d)
            hessian = pysgpp.InterpolantScalarFunctionHessian(grid, coeffs)
            

            if optimizer_choice == "adaptive_gradient_descent":
                optimizer = pysgpp.OptAdaptiveGradientDescent(ft, ftGradient)
            elif optimizer_choice == "adaptive_newton":
                optimizer = pysgpp.OptAdaptiveNewton(ft, hessian)
            elif optimizer_choice == "bfgs":
                optimizer = pysgpp.OptBFGS(ft, ftGradient)
            elif optimizer_choice == "cmaes":
                optimizer = pysgpp.OptCMAES(ft, 100)
            elif optimizer_choice == "differential_evolution":
                optimizer = pysgpp.OptDifferentialEvolution(ft)
            elif optimizer_choice == "gradient_descent":
                optimizer = pysgpp.OptGradientDescent(ft, ftGradient)
            elif optimizer_choice == "":
                optimizer = pysgpp.OptMultiStart() # default: NelderMead 
            elif optimizer_choice == "nlcg":
                optimizer = pysgpp.OptNLCG(ft, ftGradient)
            elif optimizer_choice == "nelder_mead":
                optimizer = pysgpp.OptNelderMead(ft)
            elif optimizer_choice == "newton":
                optimizer = pysgpp.OptNewton(ft, hessian)
            elif optimizer_choice == "rprop":
                optimizer = pysgpp.OptRprop(ft, ftGradient)
            else:
                print("Please specify optimizer!")
                sys.exit(1)
            
            ##################### find point with minimal loss (which are already evaluated) #################

            # find point with smallest value as start point for gradient descent
            x0Index = 0
            fX0 = functionValues[0]
            for i in range(1, len(functionValues)):
                if functionValues[i] < fX0:
                    fX0 = functionValues[i]
                    x0Index = i

            x0 = gridStorage.getCoordinates(gridStorage.getPoint(x0Index))
            ftX0 = ft.eval(x0)

            if self.verbosity > 0:
                print("\nOptimal hyperparameters so far:")
                i = 0
                for key in self.hyperparameterspace.keys():
                    if self.hyperparameterspace[key][0] == "list":
                        index = int(x0[i]*(len(self.hyperparameterspace_processed[key])-2))
                        print(key + ": " + str(self.hyperparameterspace_processed[key][index+1]))
                    else:
                        print(key + ": " + str(from_standard(self.hyperparameterspace_processed[key][0], self.hyperparameterspace_processed[key][1], x0[i])))
                    i += 1

                print("Resulting loss:")
                print(ftX0)

            ################################## Optimize with gradient descent ##################################

            # apply the gradient method and print the results.
            optimizer.setStartingPoint(x0)
            optimizer.optimize()
            xOpt = optimizer.getOptimalPoint()
            ftXOpt = optimizer.getOptimalValue()

            fXOpt = f.eval(xOpt)
            if self.verbosity > 0:
                # print(xOpt)
                print("\nOptimal hyperparameters after optimization:")
                i = 0
                for key in self.hyperparameterspace.keys():
                    if self.hyperparameterspace[key][0] == "list":
                        index = int(xOpt[i]*(len(self.hyperparameterspace_processed[key])-2))
                        print(key + ": " + str(self.hyperparameterspace_processed[key][index+1]))
                    else:
                        print(key + ": " + str(from_standard(self.hyperparameterspace_processed[key][0], self.hyperparameterspace_processed[key][1], xOpt[i])))
                    i += 1
                print("Resulting loss (Optimal value from optimization):")
                print(ftXOpt)
                print("Resulting loss (Optimal point evaluated):")
                print(fXOpt)
            
            x0_vec = []
            xOpt_vec = []
            for i in range(len(x0)):
                x0_vec.append(x0[i])
                xOpt_vec.append(xOpt[i])

            return x0_vec, xOpt_vec
        else:
            AssertionError("Type not specified correctly")




"""
    Optimization class 
    Params:
        dataset:                data of the class Dataset
        model:                  model to find the best params for (can be model or blackbox function depending on type)
        hyperparameterspace:    definition of hyperparameter space (dict with "list" or "interval" as first element of list)
        budget:                 upper bound for number of model evaluations
        verbosity:              verbosity for output
"""
class Optimization:
    def __init__(self, 
                 dataset: Dataset, 
                 model, 
                 hyperparameterspace: dict, 
                 budget: int = 100, 
                 verbosity: int = 1):
        

        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.budget = budget
        self.verbosity = verbosity


    def fit(self):
        pass



"""
    Grid Search Optimization class 
    Params:
        cv:             k-fold crossvalidation parameter
        scoring:        metric for evaluation
"""
class GridSearchOptimization(Optimization):
    def __init__(self, 
                 dataset: Dataset, 
                 model, 
                 hyperparameterspace: dict, 
                 budget: int = 100, 
                 verbosity: int = 1,
                 cv: int = 5, 
                 scoring: str = 'neg_mean_squared_error'):
        
        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.budget = budget
        self.verbosity = verbosity
        self.cv = cv
        self.scoring = scoring

        param_dimension = len(self.hyperparameterspace)
        param_per_dimension = int(self.budget**(1/param_dimension))

        for key in self.hyperparameterspace.keys():
            if self.hyperparameterspace.get(key)[0] == "list":
                self.hyperparameterspace_processed.get(key).pop(0)
                self.hyperparameterspace_processed[key] = self.hyperparameterspace_processed.get(key)
            elif self.hyperparameterspace.get(key)[0] == "interval":
                param_list = []
                for i in range(param_per_dimension):
                    upper = self.hyperparameterspace.get(key)[2]
                    lower = self.hyperparameterspace.get(key)[1]
                    param_list.append(
                        (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2)
                self.hyperparameterspace_processed[key] = param_list
            elif self.hyperparameterspace.get(key)[0] == "interval-int":
                param_list = []
                for i in range(param_per_dimension):
                    upper = self.hyperparameterspace.get(key)[2]
                    lower = self.hyperparameterspace.get(key)[1]
                    param_list.append(int(
                        (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2))
                self.hyperparameterspace_processed[key] = param_list
            else:
                print("Need to specify the type of list")


    def fit(self):
        clf = GridSearchCV(self.model, self.hyperparameterspace_processed, cv=self.cv, scoring=self.scoring, error_score='raise', verbose=self.verbosity)
        X_fit = torch.cat((self.dataset.get_X_train(), self.dataset.get_X_validation()))
        Y_fit = torch.cat((self.dataset.get_Y_train(), self.dataset.get_Y_validation()))
        return clf.fit(X_fit, Y_fit)




"""
    Random Search Optimization class 
    Params:
        cv:             k-fold crossvalidation parameter
        scoring:        metric for evaluation
"""
class RandomSearchOptimization(Optimization):
    def __init__(self, 
                 dataset: Dataset, 
                 model, 
                 hyperparameterspace: dict, 
                 budget: int = 100, 
                 verbosity: int = 1,
                 cv: int = 5, 
                 scoring: str = 'neg_mean_squared_error'):
        
        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.budget = budget
        self.verbosity = verbosity
        self.cv = cv
        self.scoring = scoring

        param_dimension = len(self.hyperparameterspace)
        param_per_dimension = int(self.budget**(1/param_dimension))

        for key in self.hyperparameterspace.keys():
            if self.hyperparameterspace.get(key)[0] == "list":
                self.hyperparameterspace_processed.get(key).pop(0)
                self.hyperparameterspace_processed[key] = self.hyperparameterspace_processed.get(key)
            elif self.hyperparameterspace.get(key)[0] == "interval":
                param_list = []
                for i in range(param_per_dimension):
                    upper = self.hyperparameterspace.get(key)[2]
                    lower = self.hyperparameterspace.get(key)[1]
                    param_list.append(
                        (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2)
                self.hyperparameterspace_processed[key] = param_list
            elif self.hyperparameterspace.get(key)[0] == "interval-int":
                param_list = []
                for i in range(param_per_dimension):
                    upper = self.hyperparameterspace.get(key)[2]
                    lower = self.hyperparameterspace.get(key)[1]
                    param_list.append(int(
                        (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2))
                self.hyperparameterspace_processed[key] = param_list
            else:
                print("Need to specify the type of list")


    def fit(self):
        clf = RandomizedSearchCV(self.model, self.hyperparameterspace_processed, cv=self.cv, scoring=self.scoring, error_score='raise', verbose=self.verbosity)
        X_fit = torch.cat((self.dataset.get_X_train(), self.dataset.get_X_validation()))
        Y_fit = torch.cat((self.dataset.get_Y_train(), self.dataset.get_Y_validation()))
        return clf.fit(X_fit, Y_fit)
    

"""
    Bayesian Optimization class 
    Params:
        only the ones for the Optimization class
"""
class BayesianOptimization(Optimization):
    def __init__(self, 
                 dataset: Dataset, 
                 model, 
                 hyperparameterspace: dict, 
                 budget: int = 100, 
                 verbosity: int = 1):
        
        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.budget = budget
        self.verbosity = verbosity

        list = []
        for key in self.hyperparameterspace.keys():
            if self.hyperparameterspace.get(key)[0] != 'list':
                self.hyperparameterspace_processed.get(key).pop(0)
                list.append(self.hyperparameterspace_processed.get(key))
            else: 
                list.append([0,1])
        self.hyperparameterspace_processed = np.array(list)



    def fit(self):
        return bayesian_optimisation(self.budget, self.model, self.hyperparameterspace_processed)


"""
    Sparse Grid Search Optimization class 
    Params:
        
"""
class SparseGridSearchOptimization(Optimization):
    def __init__(self, 
                 dataset: Dataset, 
                 model, 
                 hyperparameterspace: dict, 
                 budget: int = 100, 
                 verbosity: int = 1,
                 degree: int = 2,
                 adaptivity: float = 0.95,
                 optimizer: str = "gradient_descent"):
        
        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.budget = budget
        self.verbosity = verbosity

        self.degree = degree
        self.adaptivity = adaptivity
        self.optimizer = optimizer

        for key in self.hyperparameterspace.keys():
            self.hyperparameterspace_processed.get(key).pop(0)


    def fit(self):
        f = self.model

        # dimension of domain
        d = f.getNumberOfParameters()
        # B-spline degree
        p = self.degree
        # maximal number of grid points
        N = self.budget
        # adaptivity of grid generation
        gamma = self.adaptivity
        # choice of optimizer
        optimizer_choice = self.optimizer

        grid = pysgpp.Grid.createModBsplineGrid(d, p)
        gridGen = pysgpp.OptIterativeGridGeneratorRitterNovak(
            f, grid, N, gamma)

        functionValues = gridGen.getFunctionValues()
        if not gridGen.generate():
            print("Grid generation failed, exiting.")
            sys.exit(-1)

        
        gridStorage = grid.getStorage()
        if d == 2:
            x_values = []
            y_values = []
            z_values = []
            for i in range(gridStorage.getSize()):
                gp = gridStorage.getPoint(i)
                x_values.append(gp.getStandardCoordinate(0))
                y_values.append(gp.getStandardCoordinate(1))
                z_values.append(functionValues[i])

            #if self.verbosity >= 1:
            #    plt.plot(x_values, y_values, 'bo')

            if self.verbosity >= 1:
                fig = plt.figure()
                ax = plt.axes(projection='3d')

                ax.plot_trisurf(x_values, y_values, z_values)
                plt.show()

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
        x0 = pysgpp.DataVector(d)
        hessian = pysgpp.InterpolantScalarFunctionHessian(grid, coeffs)
        

        if optimizer_choice == "adaptive_gradient_descent":
            optimizer = pysgpp.OptAdaptiveGradientDescent(ft, ftGradient)
        elif optimizer_choice == "adaptive_newton":
            optimizer = pysgpp.OptAdaptiveNewton(ft, hessian)
        elif optimizer_choice == "bfgs":
            optimizer = pysgpp.OptBFGS(ft, ftGradient)
        elif optimizer_choice == "cmaes":
            optimizer = pysgpp.OptCMAES(ft, 100)
        elif optimizer_choice == "differential_evolution":
            optimizer = pysgpp.OptDifferentialEvolution(ft)
        elif optimizer_choice == "gradient_descent":
            optimizer = pysgpp.OptGradientDescent(ft, ftGradient)
        elif optimizer_choice == "":
            optimizer = pysgpp.OptMultiStart() # default: NelderMead 
        elif optimizer_choice == "nlcg":
            optimizer = pysgpp.OptNLCG(ft, ftGradient)
        elif optimizer_choice == "nelder_mead":
            optimizer = pysgpp.OptNelderMead(ft)
        elif optimizer_choice == "newton":
            optimizer = pysgpp.OptNewton(ft, hessian)
        elif optimizer_choice == "rprop":
            optimizer = pysgpp.OptRprop(ft, ftGradient)
        else:
            print("Please specify optimizer!")
            sys.exit(1)
        
        ##################### find point with minimal loss (which are already evaluated) #################

        # find point with smallest value as start point for gradient descent
        x0Index = 0
        fX0 = functionValues[0]
        for i in range(1, len(functionValues)):
            if functionValues[i] < fX0:
                fX0 = functionValues[i]
                x0Index = i

        x0 = gridStorage.getCoordinates(gridStorage.getPoint(x0Index))
        ftX0 = ft.eval(x0)

        if self.verbosity > 0:
            print("\nOptimal hyperparameters so far:")
            i = 0
            for key in self.hyperparameterspace.keys():
                if self.hyperparameterspace[key][0] == "list":
                    index = int(x0[i]*(len(self.hyperparameterspace_processed[key])-2))
                    print(key + ": " + str(self.hyperparameterspace_processed[key][index+1]))
                else:
                    print(key + ": " + str(from_standard(self.hyperparameterspace_processed[key][0], self.hyperparameterspace_processed[key][1], x0[i])))
                i += 1

            print("Resulting loss:")
            print(ftX0)

        ################################## Optimize with gradient descent ##################################

        # apply the gradient method and print the results.
        optimizer.setStartingPoint(x0)
        optimizer.optimize()
        xOpt = optimizer.getOptimalPoint()
        ftXOpt = optimizer.getOptimalValue()

        fXOpt = f.eval(xOpt)
        if self.verbosity > 0:
            # print(xOpt)
            print("\nOptimal hyperparameters after optimization:")
            i = 0
            for key in self.hyperparameterspace.keys():
                if self.hyperparameterspace[key][0] == "list":
                    index = int(xOpt[i]*(len(self.hyperparameterspace_processed[key])-2))
                    print(key + ": " + str(self.hyperparameterspace_processed[key][index+1]))
                else:
                    print(key + ": " + str(from_standard(self.hyperparameterspace_processed[key][0], self.hyperparameterspace_processed[key][1], xOpt[i])))
                i += 1
            print("Resulting loss (Optimal value from optimization):")
            print(ftXOpt)
            print("Resulting loss (Optimal point evaluated):")
            print(fXOpt)
        
        x0_vec = []
        xOpt_vec = []
        for i in range(len(x0)):
            x0_vec.append(x0[i])
            xOpt_vec.append(xOpt[i])

        return x0_vec, xOpt_vec
    
