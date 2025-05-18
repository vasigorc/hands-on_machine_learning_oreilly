"""
Exercise 11.

Continuing with the Olivetti faces dataset, train a classifier to predict which person is represented
in each picture, and evaluate it on the validation set. Next, use _k_-means as a dimensionality reduction
tool, and train a classifier on the reduced set. Search for the number of clusters that allows the classifier
to get the best performance: what performance can you reach? What if you append the feeatures from the
reduced set to the original features (again, searching for the best number of clusters)?
"""

import numpy as np
from pandas.core.common import random_state
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from chapter_09.common.kmeans_utils import find_optimal_clusters
from chapter_09.common.data_utils import load_split_olivetti_dataset


def main():
    # Step 1. Classify using SVC - good fit for small to medium sized, but high-demensional
    # classification tasks it is important to feature scala, as SMVs are sensitive to that
    # may need to fine-tune C, gamma parameters Use PCA to maintain 95% variance
    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        load_split_olivetti_dataset()
    )

    # Dimensionality reduction preparation
    n_components = determine_optimal_pca_components(X_train)

    # Model training and hyperparameter tuning
    search_results = train_svc_with_pca(X_train, y_train, n_components)

    print("Best SVC parameters found: ", search_results.best_params_)
    print(f"Best SVC cross-validation accuracy score: {search_results.best_score_:.4f}")
    pca_model = search_results.best_estimator_

    evaluate_svc_model(pca_model, "pca_model", X_validation, y_validation)
    # Step 2. Fit and transform _k_-means on X_train into distance matrices
    (
        k_candidates,
        k_scores,
    ) = find_optimal_clusters(np.arange(30, 60, 3), X_train)

    k = k_candidates[np.argmax(k_scores)]
    print(f"Best k for max score is {k}")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    # this will yield a (n_samples, n_clusters) distance matrix
    # each value represents a sample's distance to each cluster centroid
    # if k=50, each face becomes 50-dimensional
    kmeans_features = kmeans.fit_transform(X_train)

    # Step 3. Train SVC on these matrices, search `k` via silhouette/validation accuracy OR use validation
    # set accuracy instead of silhouette score to select `k`, since labels should now be available
    svc = SVC(kernel="linear", random_state=42)
    kmeans_model = make_pipeline(StandardScaler(), svc)
    kmeans_model.fit(kmeans_features, y_train)
    evaluate_svc_model(
        kmeans_model, "kmeans_model", kmeans.transform(X_validation), y_validation
    )

    # Step 4. Combine raw pixel values + k-means distances to create enriched features that capture:
    # - Local pixel patterns (original features)
    # - Global cluster relationships (k-means distance)
    X_train_hybrid = np.hstack([X_train, kmeans_features])  # original + distances
    X_validation_hybrid = np.hstack([X_validation, kmeans.transform(X_validation)])

    hybrid_model = make_pipeline(
        StandardScaler(), SVC(kernel="linear", random_state=42)
    )
    hybrid_model.fit(X_train_hybrid, y_train)

    evaluate_svc_model(hybrid_model, "hybrid_model", X_validation_hybrid, y_validation)


def train_svc_with_pca(X_train, y_train, n_components):
    """Train SVC with PCA dimensionality reduction using randomized parameter search."""
    pipeline = make_pipeline(
        PCA(n_components=n_components), StandardScaler(), SVC(random_state=42)
    )

    # Parameter distribution for randomized search
    param_distributions = {
        "svc__C": np.logspace(-3, 3, 20),
        "svc__kernel": ["rbf", "linear", "poly"],
        "svc__gamma": [
            "scale",
            "auto",
            0.01,
            0.1,
            1,
        ],
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=10,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    return random_search


def determine_optimal_pca_components(X_train):
    """Calculate number of PCA components to retain 95% variance"""
    pca = PCA().fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print(
        f"optimal number of retained dimensions (pixels) for the Olivetti dataset (to keep variance at 95%) is {d}"
    )
    return d


def evaluate_svc_model(model, model_name, X_validation, y_validation):
    """Evaluate the best model on validation set and print results:"""
    validation_accuracy = model.score(X_validation, y_validation)
    print(f"{model_name} validation accuracy: {validation_accuracy:.4f}")
    return validation_accuracy


if __name__ == "__main__":
    main()
    """
    The program prints:

        optimal number of retained dimensions (pixels) for the Olivetti dataset (to keep variance at 95%) is 100
        Best SVC parameters found:  {'svc__kernel': 'linear', 'svc__gamma': 'scale', 'svc__C': 54.555947811685144}
        Best SVC cross-validation accuracy score: 0.9062
        pca_model validation accuracy: 0.9531
        Best k for max score is 54
        kmeans_model validation accuracy: 0.8594
        hybrid_model validation accuracy: 0.9688
    """
