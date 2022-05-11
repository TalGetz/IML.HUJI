import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    X = pd.read_csv(filename, parse_dates=[2])
    X["DayOfYear"] = pd.to_datetime(X["Date"], errors="coerce").apply(lambda x: x.dayofyear)
    # del X["Month_Day_Year"]
    return X


def Q2(data):
    new = data[data["Country"] == "Israel"]
    seaborn.scatterplot(new["DayOfYear"], new["Temp"], c=new["Year"])
    seaborn.color_palette(None, n_colors=new["Year"].unique().size)
    plt.show()
    groups = new.groupby(["Country", "Month"])["Temp"]
    vals = groups.aggregate(["std", "mean"])
    plt.scatter(range(1, 13), vals["mean"])
    plt.errorbar(range(1, 13), vals["mean"], yerr=vals["std"])
    plt.show()


def Q3(data):
    groups = data.groupby(["Country", "Month"])["Temp"]
    vals = groups.aggregate(["std", "mean"])
    cnts = vals.index.get_level_values(0).unique()
    for country in cnts:
        plt.plot(range(1, 13), vals[vals.index.get_level_values(0) == country]["mean"])
        plt.errorbar(range(1, 13), vals[vals.index.get_level_values(0) == country]["mean"],
                     yerr=vals[vals.index.get_level_values(0) == country]["std"])
    plt.legend(cnts)
    plt.show()


def Q4(data):
    data = data[data["Country"] == "Israel"]
    Y = data["Temp"]
    X = pd.DataFrame(data["DayOfYear"])
    x_train, y_train, x_test, y_test = split_train_test(X, Y)
    ks = range(1, 11)
    errs = []
    for k in ks:
        pf = PolynomialFitting(k)
        pf.fit(x_train.to_numpy().flatten(), y_train.to_numpy().flatten())
        loss = pf.loss(x_test.to_numpy().flatten(), y_test.to_numpy().flatten())
        print(loss)
        errs.append(loss)
    plt.bar(np.array(ks), [int(x) for x in errs])
    plt.show()


def Q5(data):
    isr = data[data["Country"] == "Israel"]
    Y = isr["Temp"]
    X = pd.DataFrame(isr["DayOfYear"])
    pf = PolynomialFitting(2)
    pf.fit(X.to_numpy().flatten(), Y.to_numpy().flatten())
    vals = []
    cnts = data["Country"].unique()
    for country in cnts:
        tmp = data[data["Country"] == country]
        vals.append(pf.loss(tmp["DayOfYear"], tmp["Temp"]))
    plt.bar(cnts, [int(x) for x in vals])
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    Q2(data)

    # Question 3 - Exploring differences between countries
    Q3(data)

    # Question 4 - Fitting model for different values of `k`
    Q4(data)

    # Question 5 - Evaluating fitted model on different countries
    Q5(data)
