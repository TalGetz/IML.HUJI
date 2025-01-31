from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

import matplotlib.pyplot as plt
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = pd.read_csv(filename).dropna()
    Y = X['price']
    X['renovated'] = X['yr_renovated'] != 0
    del X['price'], X['id'], X['date'], X['yr_renovated'], X['zipcode']
    return X, Y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for column in X.columns:
        p = X[column].cov(y) / (np.std(X[column]) * np.std(y))
        plt.scatter(X[column], y)
        plt.title("{} vs. Price. Pearson Corr.: {}.".format(X[column].name, p))
        plt.savefig('{}/{}.png'.format(output_path, X[column].name))
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, Y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, Y, "./folder")

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(X, Y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percentages = np.linspace(0.1, 1, 91)
    averages = []
    stds = []
    for percentage in percentages:
        res_list = []
        for i in range(10):
            percentage = percentage.round(5)
            x_tmp, y_tmp, _, _ = split_train_test(x_train, y_train, train_proportion=percentage)
            lin_reg = LinearRegression(include_intercept=False)
            lin_reg.fit(x_tmp.to_numpy(), y_tmp.to_numpy())
            l = lin_reg.loss(x_test.to_numpy(), y_test.to_numpy())
            res_list.append(l)
        averages.append(np.average(res_list))
        stds.append(np.std(res_list))
    plt.plot(percentages, averages)
    plt.errorbar(percentages, averages, yerr=stds)
    plt.show()
