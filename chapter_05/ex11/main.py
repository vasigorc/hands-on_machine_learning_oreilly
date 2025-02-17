from datetime import datetime
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from .transformers import (
    FeatureCombinationsEngineer,
    IncomeCategoriesFeatureEngineer,
    EnhancedGeographicFeatures,
)


"""
Exercise 11. Train and fine-tune an SVM regressor on the California housing dataset. You can use the
original dataset rather than the tweaked version we used in Chapter 2, which you can load using `sklearn.datasets.fetch_california_housing()`.
The targets represent hundreds of thousands of dollars. Since there are over 20000 instances, SVMs can
be slow, so for hyperparameters tuning you should use far fewer instances (e.g. 2000) to test many more
hyperparameter combinations. What is your best model's RMSE?
"""


def main():
    """
    Main function to train and evaluate an SVM regressor on California housing data.

    Expected output:
    RMSE scores: [0.53599166 0.51728913 0.52716874 0.51090787 0.52933196]
    Mean RMSE: 0.524 (+/- 0.018)
    """
    housing = fetch_california_housing(as_frame=True)

    X = housing.data  # type: ignore
    y = housing.target  # type: ignore
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Identify outliers
    iso_forest = IsolationForest(random_state=42, contamination=0.05)  # type: ignore
    outlier_pred = iso_forest.fit_predict(X_train)

    # Keep only non-outlier data
    X_train_no_outliers = X_train[outlier_pred == 1]
    y_train_no_outliers = y_train[outlier_pred == 1]

    optimal_pipeline = create_pipeline()

    scores = cross_val_score(
        optimal_pipeline,
        X_train_no_outliers,
        y_train_no_outliers,
        scoring="neg_root_mean_squared_error",
        cv=5,
    )

    # print results
    rmse_scores = -scores
    print(f"RMSE scores: {rmse_scores}")
    print(f"Mean RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std() * 2:.3f})")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            # Apply log transformations to skewed features
            (
                "log_transform",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(np.log, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                ["AveRooms", "AveBedrms", "Population"],
            ),
            # Keep original features
            ("original", StandardScaler(), ["MedInc", "HouseAge", "AveOccup"]),
            # Add engineered features
            (
                "combinations",
                Pipeline(
                    [
                        ("combinations", FeatureCombinationsEngineer()),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["AveRooms", "AveBedrms", "Population", "AveOccup"],
            ),
            # Add income buckets
            ("income_categories", IncomeCategoriesFeatureEngineer(), ["MedInc"]),
            (
                "ocean_proximity",
                EnhancedGeographicFeatures(),
                ["Latitude", "Longitude"],
            ),
        ],
        verbose_feature_names_out=False,
    )


def create_pipeline():
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("svr", SVR(kernel="rbf", C=1.6537, epsilon=0.0157, gamma=0.0546)),  # type: ignore
        ]
    )


if __name__ == "__main__":
    main()
