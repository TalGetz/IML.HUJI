from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((len(self.classes_), X.shape[1] if len(X.shape) > 1 else 1))
        self.vars_ = np.zeros(((len(self.classes_)), X.shape[1] if len(X.shape) > 1 else 1))
        self.pi_ = np.zeros(len(self.classes_))
        for i in range(len(self.classes_)):
            self.mu_[i] = 1 / np.sum([1 for a in y if a == self.classes_[i]]) * np.sum(X[y == self.classes_[i]], axis=0)
            self.pi_[i] = np.sum([1 for a in y if a == self.classes_[i]]) / X.shape[0]

        for index, k in enumerate(self.classes_):
            self.vars_[index] = 1 / np.sum([1 for a in y if a == self.classes_[index]])
            self.vars_[index] *= np.power(np.sum(X[y == k] - self.mu_[index].T, axis=0), 2)
        pass

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
        """
        return np.array([self.classes_[np.argmax(row)] for row in self.likelihood(X)])

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = np.ndarray((X.shape[0], len(self.classes_)))
        for j, sample in enumerate(X):
            for index, k in enumerate(self.classes_):
                values = np.log(self.pi_)
                for t, _ in enumerate(sample):
                    values[index] += -np.power((sample[t] - self.mu_[index][t]), 2) / 2 / np.power(self.vars_[index][t], 2)
                    values[index] -= 1 / 2 * np.log(self.vars_[index][t])
                likelihoods[j, index] = self.classes_[np.argmax(values)]
        return likelihoods

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
        return misclassification_error(self._predict(X), y)
