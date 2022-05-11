from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((len(self.classes_), X.shape[1] if len(X.shape) > 1 else 1))
        self.cov_ = np.zeros((X.shape[1] if len(X.shape) > 1 else 1, X.shape[1] if len(X.shape) > 1 else 1))
        self.pi_ = np.zeros(len(self.classes_))
        for i in range(len(self.classes_)):
            self.mu_[i] = 1 / np.sum([1 for a in y if a == self.classes_[i]]) * np.sum(X[y == self.classes_[i]], axis=0)
            self.pi_[i] = np.sum([1 for a in y if a == self.classes_[i]]) / X.shape[0]
        for i in range(X.shape[0]):
            self.cov_ += np.dot((X[i] - self.mu_[np.where(self.classes_ == y[i])]).T,
                                X[i] - self.mu_[np.where(self.classes_ == y[i])])
        self.cov_ *= 1 / (X.shape[0] - len(self.classes_))
        self._cov_inv = inv(self.cov_)

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
        a_arr = np.dot(self._cov_inv, self.mu_.T).T
        b_arr = np.log(self.pi_) - 0.5 * np.array(
            [np.dot(np.dot(self.mu_[i], self._cov_inv), self.mu_[i].T) for i in range(len(self.classes_))])
        predictions = []
        for sample in X:
            cls = self.classes_[np.argmax([np.dot(a_arr[i], sample) + b_arr[i] for i in range(len(self.classes_))])]
            predictions.append(cls)
        return np.array(predictions)

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

        sum = 0
        for j, sample in enumerate(X):
            for index, k in enumerate(self.classes_):
                likelihoods[j,index] = self.pi_[index] * 1/np.sqrt(np.power(2 * np.pi, self.cov_.shape[0]) * np.linalg.det(self.cov_))
                likelihoods[j,index] *= np.exp(np.dot((sample - self.mu_[index]) , np.dot(np.linalg.inv(self.cov_) , (sample - self.mu_[index]).T)))

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
