import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import timeit

import itertools

import random

import copy

import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.utils import shuffle

import operator

import math
import sys
from IPython.display import clear_output

import pysgpp
from sklearn.impute import SimpleImputer

import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from openml import tasks, datasets

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor


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
    """ Dataset class
    Encapsulates the data for the hyperparameter optimization
    Arguments:
    ----------
        X: array-like
            Input data of the neural network
        Y: array-like
            Target values to the corresponding input data
        ratio: float
            Ratio to split the data into training, validation, and test set
        task_id: string
            Id of the task of OpenML
    """

    def __init__(self, X=None, Y=None, ratio=0.8, task_id=None) -> None:
        self.input_dim = 0

        if not (task_id == None):
            task = tasks.get_task(task_id)
            dataset = task.get_dataset()

            # Get the data itself as an array
            data, target, categorical_indicator, _ = dataset.get_data(
                dataset.default_target_attribute, dataset_format="array")

            self.categorical_indicator = categorical_indicator

            X = torch.Tensor(data)
            Y = torch.Tensor(target.reshape(-1, 1))

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

        self.X_train = torch.Tensor(X[:int(len(X) * ratio)])
        self.X_validation = torch.Tensor(
            self.X_train[int(len(self.X_train) * ratio):])
        self.X_train = torch.Tensor(
            self.X_train[:int(len(self.X_train) * ratio)])
        self.X_test = torch.Tensor(X[int(len(X) * ratio):])
        self.Y_train = torch.Tensor(Y[:int(len(Y) * ratio)])
        self.Y_validation = torch.Tensor(
            self.Y_train[int(len(self.Y_train) * ratio):])
        self.Y_train = torch.Tensor(
            self.Y_train[:int(len(self.Y_train) * ratio)])
        self.Y_test = torch.Tensor(Y[int(len(Y) * ratio):])

    def get_input_dim(self):
        return self.input_dim

    def get_categorical_indicator(self):
        return self.categorical_indicator

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
                               bounds=(0, 10), sampling_scales=[], n_restarts=5):
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

    for _ in range(n_restarts):
        # for each hyperparameter sample according to the distribution
        starting_point = []
        for dim in range(n_params):
            if sampling_scales[dim] == "log":
                starting_point.append(random.uniform(
                    bounds[dim, 0], bounds[dim, 1]))
            elif sampling_scales[dim] == "int":
                starting_point.append(stats.randint.rvs(
                    bounds[dim, 0], bounds[dim, 1]))
            else:
                starting_point.append(stats.uniform.rvs(
                    bounds[dim, 0], bounds[dim, 1]))

        starting_point = np.array(starting_point)

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, sampling_scales, verbosity, gp_params=None, alpha=1e-3, epsilon=1e-7):
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

    first_sample = []
    for dim in range(n_params):
        if sampling_scales[dim] == "log":
            first_sample.append(random.uniform(
                bounds[dim, 0], bounds[dim, 1]))
        elif sampling_scales[dim] == "int":
            first_sample.append(stats.randint.rvs(
                bounds[dim, 0], bounds[dim, 1]))
        else:
            first_sample.append(stats.uniform.rvs(
                bounds[dim, 0], bounds[dim, 1]))

    x_list.append(first_sample)
    y_list.append(sample_loss(first_sample))

    # if x0 is None:
    #     for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
    #         x_list.append(params)
    #         y_list.append(sample_loss(params))
    # else:
    #     for params in x0:
    #         x_list.append(params)
    #         y_list.append(sample_loss(params))

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

        next_sample = sample_next_hyperparameter(
            expected_improvement, model, yp, greater_is_better=False, bounds=bounds, sampling_scales=sampling_scales, n_restarts=10)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = []
            for dim in range(n_params):
                if sampling_scales[dim] == "log":
                    next_sample.append(random.uniform(
                        bounds[dim, 0], bounds[dim, 1]))
                elif sampling_scales[dim] == "int":
                    next_sample.append(stats.randint.rvs(
                        bounds[dim, 0], bounds[dim, 1]))
                else:
                    next_sample.append(stats.uniform.rvs(
                        bounds[dim, 0], bounds[dim, 1]))

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample)
        if verbosity > 1:
            print("Current score:", cv_score)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

        endtime = timeit.default_timer()
        time += (endtime-starttime)

    if verbosity > 0:
        print("Iterations took", time, "seconds")

    result_list = []
    for i in range(len(xp)):
        result_list.append([xp[i], yp[i]])

    return result_list, (len(result_list)+1)


def to_standard(lower, upper, value):
    return (value-lower)/(upper-lower)


def from_standard(lower, upper, value):
    return value*(upper-lower)+lower


def to_standard_log(lower, upper, value):
    a = math.log10(upper/lower)
    return math.log10(value/lower) / a


def from_standard_log(lower, upper, value):
    a = math.log10(upper/lower)
    return lower * 10**(a*value)


class Optimization:
    """ Optimization class 
    Optimization super class, main function: fit
    Arguments:
    ----------
        dataset: Dataset
            data of the class Dataset
        model: Keras model or blackboxfunction or 
            model to find the best params for 
        hyperparameterspace: dict
            definition of hyperparameter space 
        budget: string
            upper bound for number of model evaluations
        verbosity: int
            verbosity for output
    """

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


class GridSearchOptimization(Optimization):
    """
        Grid Search Optimization class 
        Params:
            cv:             k-fold crossvalidation parameter
            scoring:        metric for evaluation
    """
    """ Grid Search Optimization class 
    Grid Search Optimization class, encapsulates GridSearchCV
    Additional arguments:
    ----------
        cv: int
            k-fold crossvalidation parameter
        scoring: String
            metric for evaluation
    """

    def __init__(self,
                 dataset: Dataset,
                 model,
                 hyperparameterspace: dict,
                 budget: int = 100,
                 verbosity: int = 1,
                 cv: int = 5,
                 scoring: str = 'neg_mean_absolute_error'):

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
            upper = self.hyperparameterspace.get(key)[2]
            lower = self.hyperparameterspace.get(key)[1]
            if self.hyperparameterspace.get(key)[0] == "list":
                self.hyperparameterspace_processed.get(key).pop(0)
                self.hyperparameterspace_processed[key] = self.hyperparameterspace_processed.get(
                    key)
            elif self.hyperparameterspace.get(key)[0] == "interval":
                if param_per_dimension <= 1:
                    self.hyperparameterspace_processed[key] = [(upper+lower)/2]
                else:
                    param_list = []
                    for i in range(param_per_dimension):
                        param_list.append(
                            (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2)
                    self.hyperparameterspace_processed[key] = param_list
            elif self.hyperparameterspace.get(key)[0] == "interval-int":
                if param_per_dimension <= 1:
                    self.hyperparameterspace_processed[key] = [
                        int((upper+lower)/2)]
                else:
                    param_list = []
                    for i in range(param_per_dimension):
                        param_list.append(int(
                            (lower)+i*((upper-lower)/param_per_dimension) + (upper-lower)/param_per_dimension/2))
                    self.hyperparameterspace_processed[key] = param_list
            elif self.hyperparameterspace.get(key)[0] == "interval-log":
                if param_per_dimension <= 1:
                    step = (math.log(upper)-math.log(lower))/(2)
                    self.hyperparameterspace_processed[key] = [
                        math.exp(math.log(lower) + step)]
                else:
                    param_list = []
                    step = (math.log(upper)-math.log(lower)) / \
                        (param_per_dimension+1)
                    for i in range(param_per_dimension):
                        param_list.append(
                            math.exp(math.log(lower) + (i+1) * step))
                    self.hyperparameterspace_processed[key] = param_list
            else:
                print("Need to specify the type of list")

    def fit(self):

        lists = [self.hyperparameterspace_processed[key]
                 for key in self.hyperparameterspace_processed.keys()]

        cart_product = itertools.product(*lists)

        result_list = []
        for combination in cart_product:
            result = self.model(combination)

            result_list.append([combination, result])

        cost = 1
        for key in self.hyperparameterspace_processed.keys():
            cost *= len(self.hyperparameterspace_processed.get(key))

        return result_list, cost


class RandomSearchOptimization(Optimization):
    """ Random Search Optimization class 
    Random Search Optimization class, encapsulates RandomizedSearchCV
    Additional arguments:
    ----------
        cv: int
            k-fold crossvalidation parameter
        scoring: String
            metric for evaluation
    """

    def __init__(self,
                 dataset: Dataset,
                 model,
                 hyperparameterspace: dict,
                 budget: int = 100,
                 verbosity: int = 1,
                 cv: int = 5,
                 scoring: str = 'neg_mean_absolute_error'):

        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.budget = budget
        self.verbosity = verbosity
        self.cv = cv
        self.scoring = scoring

        for key in self.hyperparameterspace.keys():
            if self.hyperparameterspace.get(key)[0] == "list":
                self.hyperparameterspace_processed.get(key).pop(0)
                self.hyperparameterspace_processed[key] = self.hyperparameterspace_processed.get(
                    key)
            elif self.hyperparameterspace.get(key)[0] == "interval":
                upper = self.hyperparameterspace.get(key)[2]
                lower = self.hyperparameterspace.get(key)[1]
                self.hyperparameterspace_processed[key] = stats.uniform(
                    lower, upper-lower)
            elif self.hyperparameterspace.get(key)[0] == "interval-int":
                upper = self.hyperparameterspace.get(key)[2]
                lower = self.hyperparameterspace.get(key)[1]
                self.hyperparameterspace_processed[key] = stats.randint(
                    lower, upper)
            elif self.hyperparameterspace.get(key)[0] == "interval-log":
                upper = self.hyperparameterspace.get(key)[2]
                lower = self.hyperparameterspace.get(key)[1]
                self.hyperparameterspace_processed[key] = loguniform(
                    lower, upper)
            else:
                print("Need to specify the type of list")

    def fit(self):

        result_list = []

        for _ in range(self.budget):
            combination = []
            for key in self.hyperparameterspace_processed.keys():
                random_number = self.hyperparameterspace_processed[key].rvs()
                combination.append(random_number)

            result_list.append([combination, self.model(combination)])

        return result_list, self.budget


class BayesianOptimization(Optimization):
    """ Bayesian Optimization class  
    Bayesian Optimization class, implements https://thuijskens.github.io/2016/12/29/bayesian-optimisation/ 
    Additional arguments:
    ----------
        no additional arguments
    """

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
        self.sampling_scales = []

        list = []
        for key in self.hyperparameterspace.keys():
            if self.hyperparameterspace.get(key)[0] == 'interval-log':
                self.hyperparameterspace_processed.get(key).pop(0)
                lower = self.hyperparameterspace_processed.get(key)[0]
                upper = self.hyperparameterspace_processed.get(key)[1]
                list.append([math.log10(lower), math.log10(upper)])
            elif self.hyperparameterspace.get(key)[0] == 'interval-int' or self.hyperparameterspace.get(key)[0] == 'interval':
                self.hyperparameterspace_processed.get(key).pop(0)
                list.append(self.hyperparameterspace_processed.get(key))
            else:
                list.append([0, 1])

            if self.hyperparameterspace.get(key)[0] == 'interval-log':
                self.sampling_scales.append("log")
            elif self.hyperparameterspace.get(key)[0] == 'interval-int':
                self.sampling_scales.append("int")
            else:
                self.sampling_scales.append("lin")

        self.hyperparameterspace_processed = np.array(list)

    def fit(self):
        return bayesian_optimisation(self.budget, self.model, self.hyperparameterspace_processed, self.sampling_scales, self.verbosity)


class SparseGridSearchOptimization(Optimization):
    """
        Sparse Grid Search Optimization class 
        Params of the sparse grid settings:
            degree
            adaptivity
            optimizer
    """
    """ Sparse Grid Search class
    Sparse Grid Search class, adaptive sparse grid is generated and then optimized
    Additional arguments:
    ----------
        Degree: int
            Degree of the B-splines on the sparse grid
        Adaptivity: float
            Adaptivity of the sparse grid
        Optimizer: String
            Type of optimizer for the final optimization
    """

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
        self.budget = budget - 2
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

        # print("Initial level of sparse grid: ", gridGen.getInitialLevel())
        gridGen.setInitialLevel(1)
        # print("Initial level changed!")

        functionValues = gridGen.getFunctionValues()
        if not gridGen.generate():
            print("Grid generation failed, exiting.")
            sys.exit(-1)

        gridStorage = grid.getStorage()
        if d == 2:
            x_values = []
            y_values = []

            x_values_interpreted = []
            y_values_interpreted = []

            z_values = []
            for i in range(gridStorage.getSize()):
                gp = gridStorage.getPoint(i)
                keys = list(self.hyperparameterspace.keys())
                if self.hyperparameterspace[keys[0]][0] == "interval" or self.hyperparameterspace[keys[0]][0] == "interval-int":
                    x_values.append(from_standard(
                        self.hyperparameterspace[keys[0]][1], self.hyperparameterspace[keys[0]][2], gp.getStandardCoordinate(0)))
                    x_values_interpreted.append(from_standard(
                        self.hyperparameterspace[keys[0]][1], self.hyperparameterspace[keys[0]][2], gp.getStandardCoordinate(0)))
                elif self.hyperparameterspace[keys[0]][0] == "interval-log":
                    x_values.append(np.log10(from_standard_log(
                        self.hyperparameterspace[keys[0]][1], self.hyperparameterspace[keys[0]][2], gp.getStandardCoordinate(0))))
                    x_values_interpreted.append(from_standard_log(
                        self.hyperparameterspace[keys[0]][1], self.hyperparameterspace[keys[0]][2], gp.getStandardCoordinate(0)))
                else:
                    x_values.append(gp.getStandardCoordinate(0))
                    x_values_interpreted.append(gp.getStandardCoordinate(0))

                if self.hyperparameterspace[keys[1]][0] == "interval" or self.hyperparameterspace[keys[1]][0] == "interval-int":
                    y_values.append(from_standard(
                        self.hyperparameterspace[keys[1]][1], self.hyperparameterspace[keys[1]][2], gp.getStandardCoordinate(1)))
                    y_values_interpreted.append(from_standard(
                        self.hyperparameterspace[keys[1]][1], self.hyperparameterspace[keys[1]][2], gp.getStandardCoordinate(1)))
                elif self.hyperparameterspace[keys[1]][0] == "interval-log":
                    y_values.append(np.log10(from_standard_log(
                        self.hyperparameterspace[keys[1]][1], self.hyperparameterspace[keys[1]][2], gp.getStandardCoordinate(1))))
                    y_values_interpreted.append(from_standard_log(
                        self.hyperparameterspace[keys[1]][1], self.hyperparameterspace[keys[1]][2], gp.getStandardCoordinate(1)))
                else:
                    y_values.append(gp.getStandardCoordinate(1))
                    y_values_interpreted.append(gp.getStandardCoordinate(1))

                z_values.append(functionValues[i])

            if self.verbosity >= 1:
                # print("########### Generated Grid: ###########")
                x0Index = 0
                fX0 = functionValues[0]
                for i in range(1, len(functionValues)):
                    if functionValues[i] < fX0:
                        fX0 = functionValues[i]
                        x0Index = i

                x0 = gridStorage.getCoordinates(gridStorage.getPoint(x0Index))

                fig = plt.figure()
                surface = plt.scatter(x_values, y_values,
                                      c=z_values, cmap='plasma')

                plt.scatter(from_standard(
                    self.hyperparameterspace[keys[0]][1], self.hyperparameterspace[keys[0]][2], x0[0]),
                    np.log10(from_standard_log(
                        self.hyperparameterspace[keys[1]][1], self.hyperparameterspace[keys[1]][2], x0[1])),
                    100,
                    c='white',
                    marker="x",
                    alpha=1)

                fig.colorbar(surface, shrink=0.8, aspect=15)

                # plt.savefig("./Sparse" + str(self.budget) + ".pgf",bbox_inches='tight' )
                plt.show()

                # fig = plt.figure()
                # ax = plt.axes(projection='3d')

                # if len(z_values) > 100:
                #     surface = ax.plot_trisurf(x_values, y_values,
                #                     z_values, cmap='plasma')
                # else:
                #     surface = ax.scatter(x_values, y_values, z_values,
                #                c=z_values, cmap='plasma')
                # plt.xlabel(list(self.hyperparameterspace.keys())[0])
                # plt.ylabel("log10(" + list(self.hyperparameterspace.keys())[1] + ")")
                # fig.colorbar(surface, shrink=0.8, aspect=15)
                # # plt.savefig("./Normal_budget"+ str(self.budget)+"adapt"+str(self.adaptivity)+".pgf",bbox_inches='tight' )
                # plt.show()

                x0Index = 0
                fX0 = functionValues[0]
                for i in range(1, len(functionValues)):
                    if functionValues[i] < fX0:
                        fX0 = functionValues[i]
                        x0Index = i

                x0 = gridStorage.getCoordinates(gridStorage.getPoint(x0Index))

                # fig = plt.figure()
                # ax = plt.axes(projection='3d')

                # if len(z_values) > 100:
                #     surface = ax.plot_trisurf(x_values, y_values,
                #                               z_values, cmap='plasma')
                # else:
                #     surface = ax.scatter(x_values, y_values, z_values,
                #                          c=z_values, cmap='plasma')
                #     ax.scatter(from_standard(
                #         self.hyperparameterspace[keys[0]][1], self.hyperparameterspace[keys[0]][2], x0[0]),
                #         np.log10(from_standard_log(
                #             self.hyperparameterspace[keys[1]][1], self.hyperparameterspace[keys[1]][2], x0[1])),
                #         fX0,
                #         c='black',
                #         alpha=1)
                # plt.xlabel(list(self.hyperparameterspace.keys())[0])
                # plt.ylabel(
                #     "log10(" + list(self.hyperparameterspace.keys())[1] + ")")
                # fig.colorbar(surface, shrink=0.8, aspect=15)
                # ax.view_init(90, 270)
                # ax.set_zticks([])
                # ax.set_xlim([self.hyperparameterspace[keys[0]][1],
                #             self.hyperparameterspace[keys[0]][2]])
                # ax.set_ylim([np.log10(self.hyperparameterspace[keys[1]][1]), np.log10(
                #     self.hyperparameterspace[keys[1]][2])])
                # # plt.savefig("./Above_budget"+ str(self.budget)+"adapt"+str(self.adaptivity)+".pgf",bbox_inches='tight' )
                # plt.show()

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

        # if optimizer_choice == "adaptive_gradient_descent":
        #     optimizer = pysgpp.OptAdaptiveGradientDescent(ft, ftGradient)
        # elif optimizer_choice == "adaptive_newton":
        #     optimizer = pysgpp.OptAdaptiveNewton(ft, hessian)
        # elif optimizer_choice == "bfgs":
        #     optimizer = pysgpp.OptBFGS(ft, ftGradient)
        # elif optimizer_choice == "cmaes":
        #     optimizer = pysgpp.OptCMAES(ft, 100)
        # elif optimizer_choice == "differential_evolution":
        #     optimizer = pysgpp.OptDifferentialEvolution(ft)
        # elif optimizer_choice == "gradient_descent":
        #     optimizer = pysgpp.OptGradientDescent(ft, ftGradient)
        # elif optimizer_choice == "":
        #     optimizer = pysgpp.OptMultiStart()  # default: NelderMead
        # elif optimizer_choice == "nlcg":
        #     optimizer = pysgpp.OptNLCG(ft, ftGradient)
        # elif optimizer_choice == "nelder_mead":
        #     optimizer = pysgpp.OptNelderMead(ft)
        # elif optimizer_choice == "newton":
        #     optimizer = pysgpp.OptNewton(ft, hessian)
        # elif optimizer_choice == "rprop":
        #     optimizer = pysgpp.OptRprop(ft, ftGradient)
        # else:
        #     print("Please specify optimizer!")
        #     sys.exit(1)

        ##################### find point with smallest f value #################

        fX0 = functionValues[0]
        x0Index = 0
        for i in range(1, len(functionValues)):
            if functionValues[i] < fX0:
                fX0 = functionValues[i]
                x0Index = i

        x0 = gridStorage.getCoordinates(gridStorage.getPoint(x0Index))

        result = [x0, fX0]

        # optimizer = pysgpp.OptAdaptiveGradientDescent(ft, ftGradient)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptAdaptiveNewton(ft, hessian)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptBFGS(ft, ftGradient)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptCMAES(ft, 100)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptDifferentialEvolution(ft)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        optimizer = pysgpp.OptGradientDescent(ft, ftGradient)
        optimizer.setStartingPoint(x0)
        optimizer.optimize()
        result.append(optimizer.getOptimalPoint())
        result.append(f.eval(optimizer.getOptimalPoint()))

        optimizer = pysgpp.OptMultiStart(ft)  # default: NelderMead
        optimizer.setStartingPoint(x0)
        optimizer.optimize()
        result.append(optimizer.getOptimalPoint())
        result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptNLCG(ft, ftGradient)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptNelderMead(ft)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptNewton(ft, hessian)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        # optimizer = pysgpp.OptRprop(ft, ftGradient)
        # optimizer.setStartingPoint(x0)
        # optimizer.optimize()
        # result.append(f.eval(optimizer.getOptimalPoint()))

        return result, len(functionValues)


class Point():

    def __init__(self,
                 coordinates: list,
                 value: float,
                 level: int,
                 number_refined: int = 0):

        self.coordinates = coordinates
        self.value = value
        self.level = level
        self.number_refined = number_refined

    def get_coordinates(self):
        return self.coordinates

    def get_value(self):
        return self.value

    def get_level(self):
        return self.level

    def get_number_refined(self):
        return self.number_refined

    def increase_number_refined(self):
        self.number_refined += 1


class IterativeRandomOptimization(Optimization):
    """
        Iterative random Search Optimization class 
        Params of the sparse grid settings:

    """
    """ Sparse Grid Search class
    Sparse Grid Search class, adaptive sparse grid is generated and then optimized
    Additional arguments:
    ----------
        Degree: int
            Degree of the B-splines on the sparse grid
        Adaptivity: float
            Adaptivity of the sparse grid
        Optimizer: String
            Type of optimizer for the final optimization
    """

    def __init__(self,
                 dataset: Dataset,
                 model,
                 hyperparameterspace: dict,
                 budget: int = 100,
                 verbosity: int = 0,
                 adaptivity: float = 0.5,
                 init_points: int = 10,
                 alternative: int = 0):

        self.dataset = dataset
        self.model = model
        self.hyperparameterspace = hyperparameterspace
        self.hyperparameterspace_processed = copy.deepcopy(hyperparameterspace)
        self.budget = budget
        self.verbosity = verbosity

        self.adaptivity = adaptivity
        self.init_points = init_points
        self.alternative = alternative

        for key in self.hyperparameterspace.keys():
            if self.hyperparameterspace.get(key)[0] == "list":
                self.hyperparameterspace_processed.get(key).pop(0)
                self.hyperparameterspace_processed[key] = self.hyperparameterspace_processed.get(
                    key)
            elif self.hyperparameterspace.get(key)[0] == "interval":
                upper = self.hyperparameterspace.get(key)[2]
                lower = self.hyperparameterspace.get(key)[1]
                self.hyperparameterspace_processed[key] = stats.uniform(
                    lower, upper)
            elif self.hyperparameterspace.get(key)[0] == "interval-int":
                upper = self.hyperparameterspace.get(key)[2]
                lower = self.hyperparameterspace.get(key)[1]
                self.hyperparameterspace_processed[key] = stats.randint(
                    lower, upper)
            elif self.hyperparameterspace.get(key)[0] == "interval-log":
                upper = self.hyperparameterspace.get(key)[2]
                lower = self.hyperparameterspace.get(key)[1]
                self.hyperparameterspace_processed[key] = loguniform(
                    lower, upper)
            else:
                print("Need to specify the type of list")

    def draw_value(self, key, lower, upper):
        if self.hyperparameterspace.get(key)[0] == "interval":
            result = random.uniform(lower, upper)
            return result
        elif self.hyperparameterspace.get(key)[0] == "interval-int":
            result = stats.randint(lower, upper+1).rvs()
            return result
        elif self.hyperparameterspace.get(key)[0] == "interval-log":
            result = loguniform(lower, upper).rvs()
            return result
        else:
            print("Need to specify the type of list")

    def draw_value_normal_distribution(self, key, lower, upper):
        if self.hyperparameterspace.get(key)[0] == "interval":
            result = np.random.normal((upper+lower)/2, (upper-lower)/2)
            #result = random.uniform(lower, upper)
            return result
        elif self.hyperparameterspace.get(key)[0] == "interval-int":
            result = int(np.random.normal((upper+lower)/2, (upper-lower)/2))
            return result
        elif self.hyperparameterspace.get(key)[0] == "interval-log":
            result = np.random.lognormal((upper+lower)/2, (upper+lower)/2)
            return result
        else:
            print("Need to specify the type of list")

    def fit(self):

        points = []

        # initialization (draw initial random points with level 1)
        for _ in range(min([self.init_points, self.budget])):
            coordinates = []
            for key in self.hyperparameterspace.keys():
                coordinates.append(self.draw_value(key, self.hyperparameterspace.get(key)[
                                   1], self.hyperparameterspace.get(key)[2]))

            value = self.model(coordinates)

            points.append(Point(coordinates=coordinates, value=value, level=0))

        points.sort(key=operator.attrgetter('value'))

        while len(points) + 2 * len(self.hyperparameterspace) <= self.budget:

            coefficients = []
            for i in range(len(points)):
                rank_i = i
                level_i = points[i].get_level()
                number_refined_i = points[i].get_number_refined()
                value_i = points[i].get_value()
                coefficients.append(
                    (rank_i + value_i + 1)**(1-self.adaptivity) * (level_i + number_refined_i + 1)**(self.adaptivity))

            index_refine = 0
            current_smallest = coefficients[0]
            for i in range(len(coefficients)):
                if coefficients[i] < current_smallest:
                    current_smallest = coefficients[i]
                    index_refine = i

            level_best = points[index_refine].get_level()
            coordinates_best = points[index_refine].get_coordinates()

            points[index_refine].increase_number_refined()

            if self.alternative == 0:
                # ################ Alternative 1 ###############
                ######## intervals in each dimension #########
                intervals = []
                keyIndex = 0
                for key in self.hyperparameterspace.keys():
                    lower = self.hyperparameterspace.get(key)[1]
                    for j in range(len(points)):
                        if lower < points[j].get_coordinates()[keyIndex] and points[j].get_coordinates()[keyIndex] < coordinates_best[keyIndex] and points[j].get_level() <= level_best:
                            lower = points[j].get_coordinates()[keyIndex]

                    upper = self.hyperparameterspace.get(key)[2]
                    for j in range(len(points)):
                        if upper > points[j].get_coordinates()[keyIndex] and points[j].get_coordinates()[keyIndex] > coordinates_best[keyIndex] and points[j].get_level() <= level_best:
                            upper = points[j].get_coordinates()[keyIndex]

                    intervals.append([lower, upper])
                    keyIndex += 1

                for _ in range(2* len(self.hyperparameterspace)):
                    coordinates = []
                    keyIndex = 0
                    for key in self.hyperparameterspace.keys():
                        coordinates.append(self.draw_value(
                            key, intervals[keyIndex][0], intervals[keyIndex][1]))
                        keyIndex += 1

                    value = self.model(coordinates)

                    points.append(Point(coordinates=coordinates,
                                        value=value, level=level_best+1))
                ################ Alternative 1 END ############
            elif self.alternative == 1:

                ################ Alternative 2 ###############
                ####### uniform sampling n-dim balls #########

                # compute all distances to other points
                distances = [math.dist(
                    points[index_refine].get_coordinates(), x.get_coordinates()) for x in points]
                highest_distance = max(distances)

                # find smallest distance (replace smallest distance zero with highest distance)
                corners = [[], []]
                for key in self.hyperparameterspace.keys():
                    corners[0].append(self.hyperparameterspace.get(key)[1])
                    corners[1].append(self.hyperparameterspace.get(key)[2])

                smallest_dist = math.dist(corners[0], corners[1])
                for i in range(len(points)):

                    if distances[i] == 0:
                        distances[i] = highest_distance

                    if i != index_refine and distances[i] < smallest_dist and points[i].get_level() <= points[index_refine].get_level():
                        smallest_dist = distances[i]

                # smallest_dist = (highest_distance + smallest_dist) / ((points[index_refine].get_level()+1)*2)
                smallest_dist = (highest_distance + smallest_dist) / ((points[index_refine].get_level()+2)*2)

                # smallest_dist = sum(distances) / len(distances)

                # sample 2*dim new points in this ball
                for _ in range(2 * len(self.hyperparameterspace)):
                    # an array of d normally distributed random variables
                    u = np.random.normal(
                        0, 1, len(self.hyperparameterspace))
                    norm = np.sum(u**2) ** (0.5)
                    r = random.uniform(
                         0, 1)**(1.0/len(self.hyperparameterspace))
                    # r = np.random.normal(0, smallest_dist)**(1.0/len(self.hyperparameterspace))

                    x = r*u/norm
                    x = x*smallest_dist
                    coordinates = []
                    # points[index_refine].get_coordinates()

                    keyIndex = 0
                    for key in self.hyperparameterspace.keys():
                        coordinates.append(points[index_refine].get_coordinates()[
                                           keyIndex] + x[keyIndex])

                        coordinates[keyIndex] = max(
                            [coordinates[keyIndex], self.hyperparameterspace.get(key)[1]])
                        coordinates[keyIndex] = min(
                            [coordinates[keyIndex], self.hyperparameterspace.get(key)[2]])

                        keyIndex += 1

                    value = self.model(coordinates)
                    points.append(Point(coordinates=coordinates,
                                  value=value, level=level_best+1))
                # ################ Alternative 2 END ############
            if self.alternative == 2:
                # ################ Alternative 3 ###############
                ######## intervals in each dimension, but not uniform prob #########
                # compute all distances to other points
                distances = [math.dist(
                    points[index_refine].get_coordinates(), x.get_coordinates()) for x in points]
                highest_distance = max(distances)

                # find smallest distance (replace smallest distance zero with highest distance)
                corners = [[], []]
                for key in self.hyperparameterspace.keys():
                    corners[0].append(self.hyperparameterspace.get(key)[1])
                    corners[1].append(self.hyperparameterspace.get(key)[2])

                smallest_dist = math.dist(corners[0], corners[1])
                for i in range(len(points)):

                    if distances[i] == 0:
                        distances[i] = highest_distance

                    if i != index_refine and distances[i] < smallest_dist and points[i].get_level() <= points[index_refine].get_level():
                        smallest_dist = distances[i]

                # smallest_dist = (highest_distance + smallest_dist) / ((points[index_refine].get_level()+1)*2)
                smallest_dist = 1# (highest_distance + smallest_dist) / ((points[index_refine].get_level()+2)*4)

                for _ in range(2 * len(self.hyperparameterspace)):
                    coordinates = []
                    keyIndex = 0
                    for key in self.hyperparameterspace.keys():
                        lower = max([points[index_refine].get_coordinates()[keyIndex]-smallest_dist, self.hyperparameterspace.get(key)[1]])
                        lower = min([lower, self.hyperparameterspace.get(key)[2]])
                        
                        upper = min([points[index_refine].get_coordinates()[keyIndex]+smallest_dist, self.hyperparameterspace.get(key)[2]])
                        upper = max([upper, self.hyperparameterspace.get(key)[1]])

                        if upper - lower < 0:
                            print(lower, upper)
                            print(smallest_dist)
                        
                        coordinates.append(self.draw_value_normal_distribution(
                            key, lower, upper))
                        keyIndex += 1

                    value = self.model(coordinates)

                    points.append(Point(coordinates=coordinates,
                                        value=value, level=level_best+1))
            points.sort(key=operator.attrgetter('value'))

        return points
