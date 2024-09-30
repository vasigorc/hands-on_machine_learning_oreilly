from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    TransformerMixin,
    check_is_fitted,
    clone,
)

"""
From the solution exercise:
Rather than restrict ourselves to k-Nearest Neighbors regressors, let's create a transformer
that accepts any regressor. For this, we can extend the MetaEstimatorMixin and have a required
estimator argument in the constructor.
"""


class TransformerFromRegressor(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    """
    The fit() method must work on a clone of this estimator, and it must also save
    feature_names_in_. The MetaEstimatorMixin will ensure that estimator is listed as a
    required parameters, and it will update get_params() and set_params() to make the
    estimator's hyperparameters available for tuning.
    """

    def fit(self, X, y=None):
        estimator_ = clone(self.estimator)
        estimator_.fit(X, y)

        """
        store the fitted estimator as an attribute of the TransformerFromRegressor instance
        So that transform() method can access the fitted estimator later
        """
        self.estimator_ = estimator_
        """
        n_features_in_ is an attribute commonly used in scikit-learn estimators to store the
        number of features that the estimator was trained on
        """
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator, "feature_names_in_"):
            self.feature_names_in_ = self.estimator.feature_names_in_
        return self

    def transform(self, X):
        # check if the estimator has attributes ending with '_', i.e. fitted attributes
        check_is_fitted(self)
        predictions = self.estimator_.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_feature_names_out(self, names=None):
        check_is_fitted(self)
        n_outputs = getattr(self.estimator_, "n_outputs_", 1)
        estimator_class_name = self.estimator_.__class__.__name__
        estimator_short_name = estimator_class_name.lower().replace("_", "")
        return [f"{estimator_short_name}_prediction_{i}" for i in range(n_outputs)]
