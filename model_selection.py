import vowpalwabbit
import math
import numpy as np
import pandas as pd
from preprocessing import *
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

def generate_grid(rank:np.array, l2:np.array, lrate:np.array, passes:np.array) -> iter:
    """
    Brute-force hyperparameter grid generation.
    :return: iterable of hyperparameters
    """
    param_grid = {"rank": rank, "l2": l2,
                 "lrate": lrate, "passes": passes}
    grid = ParameterGrid(param_grid)
    return iter(grid)

def fit(model, training):
    """Train the model"""
    for example in training:
        model.learn(example)
    model.finish()

def pred(model, validation) -> list:
    """Returns a list of predictions given some validation dataframe"""
    return [model.predict(example) for example in validation]

def create_model(hyperparams:dict, train, validation, validation_df, r_model:bool=False):
    """
    Creates the model, we're only varying rank, the l2 param, the learning rate during gradient descent and the number
    of epochs per training example.
    :param hyperparams: dictionary containing the following entries: rank, l2, lrate, passes
    :param r_model: whether we return the model we created
    return: model details
    """
    lrate, l2, rank, passes = (float(hyperparams["lrate"]), float(hyperparams["l2"]), int(hyperparams["rank"]),
                               int(hyperparams["passes"]))
    model = vowpalwabbit.Workspace(q=['um'], rank=rank, l2=l2, l=lrate, passes=passes,
                                   arg_str='--decay_learning_rate 0.99 --power_t 0 --cache_file model.cache',
                                   P=1000,  quiet=True, enable_logging=r_model)
    fit(model, train)
    predictions = pred(model, validation)
    rmse = math.sqrt(mean_squared_error(validation_df['rating'], predictions))
    if r_model:
        return model, rmse, hyperparams
    return rmse, hyperparams

