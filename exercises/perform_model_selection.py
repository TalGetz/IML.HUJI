from __future__ import annotations
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

datasize = 470.0


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.random.uniform(-1.2, 2, n_samples)
    y_noise = np.random.normal(0, np.sqrt(noise), n_samples)
    y = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    noisy_y = y + y_noise
    train_x, train_y, test_x, test_y = split_train_test(DataFrame(x), noisy_y, 2.0 / 3.0)
    plt.scatter(train_x, train_y)
    plt.scatter(test_x, test_y)
    x_sorted, y_sorted = [v[0] for v in sorted(zip(x, y))], [v[1] for v in sorted(zip(x, y))]
    plt.plot(x_sorted, y_sorted)
    plt.legend(["true", "train", "test"])
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    ks = list(range(11))
    train_errors = []
    validation_errors = []
    for k in ks:
        train_error, validation_error = cross_validate(PolynomialFitting(k), train_x.to_numpy().flatten(),
                                                       train_y.to_numpy(), scoring=mean_square_error)
        train_errors.append(train_error)
        validation_errors.append(validation_error)

    plt.plot(ks, train_errors)
    plt.plot(ks, validation_errors)
    plt.legend(["train errors", "validation errors"])
    plt.title("Errors vs K")
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = ks[validation_errors.index(min(validation_errors))]
    model = PolynomialFitting(k_star).fit(train_x.to_numpy().flatten(), train_y.to_numpy())
    test_error = model.loss(train_x.to_numpy().flatten(), train_y.to_numpy())
    print(
        f"best k value: {k_star}, test error: {test_error}, validation error: {validation_errors[validation_errors.index(min(validation_errors))]}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    perc = n_samples / datasize
    train_x, train_y, test_x, test_y = split_train_test(X, y, perc)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    lambdas = list(np.linspace(1e-5, 1, n_evaluations))
    ridge_train_errors = []
    ridge_validation_errors = []
    lasso_train_errors = []
    lasso_validation_errors = []
    for lamb in lambdas:
        train_error, validation_error = cross_validate(RidgeRegression(lamb), train_x.to_numpy(), train_y.to_numpy(),
                                                       scoring=mean_square_error)
        ridge_train_errors.append(train_error)
        ridge_validation_errors.append(validation_error)
        train_error, validation_error = cross_validate(Lasso(lamb, max_iter=5000), train_x.to_numpy(),
                                                       train_y.to_numpy(),
                                                       scoring=mean_square_error)
        lasso_train_errors.append(train_error)
        lasso_validation_errors.append(validation_error)

    plt.plot(lambdas, ridge_train_errors)
    plt.plot(lambdas, ridge_validation_errors)
    plt.plot(lambdas, lasso_train_errors)
    plt.plot(lambdas, lasso_validation_errors)
    plt.legend(["ridge train errors", "ridge validation errors", "lasso train errors", "lasso validation errors"])
    plt.title("Errors vs Lambdas")
    plt.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = np.argmin(ridge_validation_errors)
    best_lasso = np.argmin(lasso_validation_errors)
    l_ridge = lambdas[best_ridge]
    l_lasso = lambdas[best_lasso]
    ridge_err = RidgeRegression(l_ridge).fit(train_x.to_numpy(), train_y.to_numpy()).loss(test_x.to_numpy(), test_y.to_numpy())
    lasso_err = mean_square_error(test_y.to_numpy(), Lasso(l_lasso).fit(train_x, train_y).predict(test_x))
    linreg_err = LinearRegression().fit(train_x.to_numpy(), train_y.to_numpy()).loss(test_x.to_numpy(), test_y.to_numpy())

    print(f"Ridge Lambda: {l_ridge}\nLasso Lambda: {l_lasso}")
    print(f"Ridge Error: {ridge_err}\nLasso Error: {lasso_err}\nLeast Squares Error: {linreg_err}")


def part1():
    np.random.seed(0)
    select_polynomial_degree()
    # Q4
    select_polynomial_degree(noise=0)
    # Q5
    select_polynomial_degree(n_samples=1500, noise=10)


if __name__ == '__main__':
    np.random.seed(0)
    part1()
    select_regularization_parameter()
