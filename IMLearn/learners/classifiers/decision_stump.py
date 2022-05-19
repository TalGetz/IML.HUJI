from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product

from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self):
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_stats = None
        best_err = np.inf
        for sign in [1, -1]:
            for j in range(X.shape[1]):
                thr, thr_err = self._find_threshold(X[:, j], y, sign)
                if thr_err <= best_err:
                    best_stats = thr, j, sign
                    best_err = thr_err
        self.threshold_, self.j_, self.sign_ = best_stats

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        responses = ((X[:, self.j_] >= self.threshold_) - 0.5) * 2 * self.sign_
        return responses

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        zipped = sorted(zip(values, labels), reverse=True)
        values, labels = [x[0] for x in zipped], [x[1] for x in zipped]
        thresholds = values
        best_threshold = thresholds[0]
        response = ((values >= thresholds[0]) - 0.5) * 2 * sign
        err = best_err = np.sum([abs(y_p) for y_t, y_p in zip(response, labels) if y_t != my_sign(y_p)])
        for i, threshold in enumerate(thresholds):
            if i == 0:
                continue
            response[i] = sign
            err -= ((int(my_sign(labels[i]) == sign) - 0.5) * 2) * abs(labels[i])
            if best_err >= err:
                best_threshold = threshold
                best_err = err
        return best_threshold, best_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(self.predict(X), y)


def my_sign(x):
    return np.sign(x) if x != 0 else 1