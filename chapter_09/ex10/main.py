"""
Exercise 10.

The classic Olivetti faces dataset contains 400 grayscale 64 X 64-pixel images of faces. Each image
is flattened to a 1D vector of size 4096. Forty different people were photographed (10 times each),
and the usual task is to train a model that can predict which person is represented in each picture.
Loadthe dataset using the `sklearn.datasets.fetch_olivetti_faces()` function, then split it into a training
set, a validation set, and a test set (note that the dataset is already scaled between 0 and 1). Since
the dataset is quite small, you will probably want to use stratified sampling to ensure that there are
the same number of images per person in each set. Next, cluster the images using _k_-means, and ensure
that you have a good number of clusters (using one of the techniques discussed in this chapter). Visualize
the clusters: do you see similar faces in each cluster?
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from chapter_09.common.data_utils import load_split_olivetti_dataset
from chapter_09.common.kmeans_utils import find_optimal_clusters


def main():
    # Step 1. Load and split the dataset
    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        load_split_olivetti_dataset()
    )

    # Step 2: Find optimal clusters (simplified version)
    (
        k_candidates,
        k_scores,
    ) = find_optimal_clusters(np.arange(30, 60, 3), X_train)

    k = k_candidates[np.argmax(k_scores)]
    print(f"Best k for max score is {k}")

    # Step 3. Cluster the images using the optimal number of clusters
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_train)
    train_cluster_labels = kmeans.labels_

    # Step 4. Visualize images
    # after splitting, reshape the training data into 64x64 images
    train_images = X_train.reshape(-1, 64, 64)
    five_random_clusters = np.random.choice(k, size=5, replace=False)
    save_clusters_summary(
        cluster_ids=five_random_clusters,
        images=train_images,
        labels=train_cluster_labels,
    )


def save_clusters_summary(
    cluster_ids,
    images,
    labels,
    n_samples=5,
    figsize=(15, 8),
    filename="clusters_summary.png",
):
    n_clusters = len(cluster_ids)
    plt.figure(figsize=figsize)

    for i, cluster_id in enumerate(cluster_ids):
        cluster_mask = labels == cluster_id
        cluster_images = images[cluster_mask]
        cluster_size = len(cluster_images)

        # Add cluster metadata text
        plt.figtext(
            0.05,
            0.9 - (i * 0.18),  # Adjust positioning as needed
            f"Cluster {cluster_id}\nSize: {cluster_size}",
            ha="left",
            va="top",
            fontsize=12,
        )

        # Plot sample images
        for j in range(n_samples):
            if j >= cluster_size:  # Handle clusters smaller than n_samples
                break
            ax = plt.subplot(n_clusters, n_samples, i * n_samples + j + 1)
            ax.imshow(cluster_images[j], cmap="gray")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
