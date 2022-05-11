from math import atan2, pi

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt

from IMLearn.metrics import accuracy

pio.templates.default = "simple_white"
import pandas as pd


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        data = np.load(f)
        X = data[:, [0, 1]]
        Y = data[:, 2]

        def callback(fit: Perceptron, x: np.ndarray, y: int):
            fit.training_loss_.append(fit._loss(X, Y))

        # Fit Perceptron and record loss in each fit iteration
        p = Perceptron(callback=callback)
        p.fit(X, Y)

        # Plot figure
        plt.plot(p.training_loss_)
        plt.title(n)
        plt.xlabel("Iteration")
        plt.ylabel("Training Loss")
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
        # Load dataset
        data = np.load(f)
        X = data[:, [0, 1]]
        Y = data[:, 2]

        lda = LDA()

        lda.fit(X, Y)

        lda_pred = lda.predict(X)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, Y)
        gnb_pred = gnb.predict(X)

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"{f} LDA. Accuracy {accuracy(Y, lda_pred)}", f"{f} Gaussian Naive Bayes. Accuracy {accuracy(Y, gnb_pred)}"))
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=lda_pred, symbol=Y)), row=1,
                      col=1)
        for i in range(lda.mu_.shape[0]):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=1)

        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=gnb_pred, symbol=Y)), row=1,
                      col=2)
        for i in range(gnb.mu_.shape[0]):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=2)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
