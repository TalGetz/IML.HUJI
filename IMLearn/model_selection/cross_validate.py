from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    data = list(zip(X, y))
    amount_of_data_per_block = y.shape[0] / cv
    train_scores = []
    test_scores = []
    blocks = [data[int(index * amount_of_data_per_block): int((index + 1) * amount_of_data_per_block)
    if int((index + 1) * amount_of_data_per_block) < len(data) else -1] for index in range(cv)]
    for index, block in enumerate(blocks):
        train_data = [i for sub in blocks if sub is not block for i in sub]
        test_data = block
        train_x = np.array([xy[0] for xy in train_data])
        train_y = np.array([xy[1] for xy in train_data])
        test_x = np.array([xy[0] for xy in test_data])
        test_y = np.array([xy[1] for xy in test_data])
        estimator.fit(train_x, train_y)
        train_scores.append(scoring(train_y, estimator.predict(train_x)))
        test_scores.append(scoring(test_y, estimator.predict(test_x)))
    return np.average(train_scores), np.average(test_scores)
