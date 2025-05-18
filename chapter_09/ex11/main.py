"""
Exercise 11.

Continuing with the Olivetti faces dataset, train a classifier to predict which person is represented
in each picture, and evaluate it on the validation set. Next, use _k_-means as a dimensionality reduction
tool, and train a classifier on the reduced set. Search for the number of clusters that allows the classifier
to get the best performance: what performance can you reach? What if you append the feeatures from the
reduced set to the original features (again, searching for the best number of clusters)?
"""

from chapter_09.common.data_utils import load_split_olivetti_dataset
from sklearn.decomposition import PCA
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV


def main():
    # Step 1. Classify using SVC - good fit for small to medium sized, but high-demensional
    # classification tasks it is important to feature scala, as SMVs are sensitive to that
    # may need to fine-tune C, gamma parameters
    # Use PCA to maintain 95% variance
    # Step 2. Fit and transform _k_-means on X_train into distance matrices
    # Step 3. Train SVC on these matrices, search `k` via silhouette/validation accuracy OR use validation
    # set accuracy instead of silhouette score to select `k`, since labels should now be available

    # Step 1. Classify using SVC - good fit for small to medium sized, but high-demensional
    # classification tasks it is important to feature scala, as SMVs are sensitive to that
    # may need to fine-tune C, gamma parameters Use PCA to maintain 95% variance
    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        load_split_olivetti_dataset()
    )
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print(
        f"optimal number of retained dimensions (pixels) for the Olivetti dataset (to keep variance at 95%) is {d}"
    )
    # obtain the reduced training set
    pca = PCA(n_components=d)
    X_train_reduced = pca.fit_transform(X_train)

    param_distributions = {
        "C": np.logspace(-3, 3, 20),
        "kernel": ["rbf", "linear", "poly"],
        "gamma": [
            "scale",
            "auto",
            0.01,
            0.1,
            1,
        ],
    }
    svc = SVC(random_state=42)
    scoring_type = "accuracy"
    random_search = RandomizedSearchCV(
        svc,
        param_distributions,
        n_iter=10,
        cv=3,
        scoring=scoring_type,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    print("Best SVC parameters found: ", random_search.best_params_)
    print(
        f"Best SVC cross-validation {scoring_type} score: {random_search.best_score_:.4f}"
    )


if __name__ == "__main__":
    main()
