import numpy as np
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    check_array,
    check_is_fitted,
)

"""
Exercise 6

Try to implement the `StandardScalerClone` class again from scratch, 
- then add support for the `inverse_transform()` method: executing 
  `scaler.inverse_transform(scaler.fit_transform(X))` should 
return an array very close to `X`. 
- Then add support for feature names: set `feature_names_in_` in 
  the `fit` method if the input is a DataFrame. This attribute should be a NumPy array of column names. 
- Lastly, implement the `get_features_names_out()` method: it should have one optional 
  `input_features=None`. If passed, the method should check that is length matches `n_features_in_`, 
  and it should match `feature_names_in_` if it is defined; then `input_features` should be returned. 
  If `input_features` is `None`, then the method should either return `feature_names_in_` if it is 
  defined or `np.array(["x0", "x1", ...])` with length `n_features_in_` otherwise.
"""


class StandardScalerClone(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, copy=True, with_mean=True):
        self.with_mean = with_mean
        self.copy = copy

    def fit(self, X, y=None):
        X = check_array(X)  # check that X is an array with finite float values
        X_orig = X
        # axis=0 means that standard deviation is calculated along the rows (for every feature/column),
        # creating thus a 1D array
        # where each element is the standard deviation of a particular feature
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # get a count of feature. Every estimator stores this in fit()
        self.n_features_in_ = X.shape[1]
        if hasattr(X_orig, "columns"):
            self.n_features_names_in_ = np.array(X_orig.columns, dtype=object)
        return self

    def _check_input(self, X):
        # looks for learned attributes (those with trainiling _ e.g. mean_)
        check_is_fitted(self)
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("Unexpected number of features!")
        return X

    def transform(self, X):
        X = self._check_input(X)
        if self.with_mean:
            # center the data by substracting the mean of each feaure for all the values
            # this shifts the distribution so that it's mean becomes 0
            # the ultimate purpose is to put all features on a similar scale
            X = X - self.mean_
        # standardize the scale of all features
        return X / self.scale_

    def inverse_transform(self, X):
        X = self._check_input(X)
        """
        scale the features back to what they were before transform. This may be useful
        in order to interpret results at the original scale or if we want to
        compare with the predictions at the original scale.

        Note that `scale_` and `mean_` operations are applied in reverse order. This is
        because these operations are not commutative
        """
        X = X * self.scale_
        # Python's ternary operator
        return X + self.mean_ if self.with_mean else X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            # the third parameter is the default value that will be returned
            return getattr(
                self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)]
            )
        else:
            if len(input_features) != self.n_features_in_:
                raise ValueError("Invalid number of features!")
            # not np.all - checks if any element is False
            if hasattr(self, "feature_names_in_") and not np.all(
                self.n_features_names_in_ == input_features
            ):
                raise ValueError("input_features != feature_names_in_")
            return input_features
