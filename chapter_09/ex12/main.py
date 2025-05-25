"""
Exercise 12.

Train a Gaussian mixture model on the Olivetti faces dataset. To speed up the algorithm, you should
probably reduce the dataset's dimensionality (e.g. use PCA, preserving 99% of variance). Use the model
to generate some new faces (using the `sample()` method), and visualize them (if you used PCA, you will
need to use its `inverse_transform()` method). Try to modify some images (e.g. rotate, flip, darken)
and see if the model can detect the anomalies (i.e. compare the output of the `score_samples()` method
for normal images and for anomalies).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from chapter_09.common.clustering_utils import compute_aic, find_optimal_clusters
from chapter_09.common.dimensionality_reduction_utils import (
    determine_optimal_pca_components,
)


def main():
    # Step 1. Split the dataset into training and test sets
    olivetti = fetch_olivetti_faces()
    X_train, X_test, y_train, y_test = train_test_split(
        olivetti.data,
        olivetti.target,
        stratify=olivetti.target,
        test_size=0.2,
        random_state=42,
    )

    # Step 2. Reduce dataset's dimensionality with PCA
    # establishes as 222
    n_components = determine_optimal_pca_components(X_train, 0.99)
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train)

    # Step 3. Select the number of cluster using aic() or `BayesianGaussianMixture`
    (k_candidates, k_scores) = find_optimal_clusters(
        np.arange(30, 60, 3), X_train_reduced, compute_aic
    )
    optimal_k = k_candidates[np.argmax(k_scores)]
    # establishes as 57
    print(f"Best k for max score is {optimal_k}")

    # Step 4. Train a Gaussian mixture on the reduced dataset
    gm = GaussianMixture(n_components=optimal_k, n_init=10, random_state=42)
    gm.fit(X_train_reduced)
    print(f"GM converged {gm.converged_} in {gm.n_iter_} iterations")

    # Step 5. Sample a few new faces and visualize them
    X_new_reduced, y_new_reduced = gm.sample(4)
    X_new = pca.inverse_transform(X_new_reduced)
    # reshape the data into 64x64 images
    new_images = X_new.reshape(-1, 64, 64)

    # Display sampled faces
    _, axes = plt.subplots(1, 4, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(new_images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("chapter_09/ex12/generated_images.png")
    plt.close()

    # Step 6. Modify some other images and see if the model can detect anomalies
    # For this task we want to use unseen data: pristine normal data, but not the
    # data that the model was trained on.

    densities = gm.score_samples(X_train_reduced)
    density_threshold = np.percentile(densities, 5)

    normal_test = X_test[:100]
    anomalous_test = create_anomalies(normal_test.copy())

    # Calculate anomaly rates
    normal_scores = gm.score_samples(pca.transform(normal_test))
    anomaly_scores = gm.score_samples(pca.transform(anomalous_test))

    normal_anomaly_rate = np.mean(normal_scores < density_threshold)
    true_anomaly_rate = np.mean(anomaly_scores < density_threshold)

    print("\nAnomaly Detection Performance:")
    print(f"- False positive rate: {normal_anomaly_rate:.1f}%")
    print(f"- True detection rate: {true_anomaly_rate:.1f}%")


def create_anomalies(images):
    """Apply random modifications to images"""
    modified = []
    for img in images.reshape(-1, 64, 64):
        # Random modification
        modification = np.random.choice(["flip", "rotate", "darken"])
        if modification == "flip":
            modified.append(np.fliplr(img))
        elif modification == "rotate":
            modified.append(np.rot90(img, k=np.random.randint(1, 4)))
        else:
            modified.append(img * np.random.uniform(0.2, 0.6))  # Darken
    return np.array(modified).reshape(len(images), -1)


if __name__ == "__main__":
    main()
