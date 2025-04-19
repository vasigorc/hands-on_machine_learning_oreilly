"""
Exercise 10.

Use t-SNE to reduce the first 5000 images of the MNIST dataset down to 2 dimensions and
plot the result using Matplotlib. You can use a scatterplot using 10 different colors to represent each
image's target class. Alternatively, you can replace each dot in the scatterplot with the corresponding
instance's class (a digit from 0 to 9), or even plot scaled-dwon versions of the digit images themselves
(if you plot all digits the visualization will be too cluttered, so you should either draw a random
sample or plot an instance only if no other instance has already been plotted at a close distance).
You should get a nice visualization with well-separated clusters of digits. Try using other dimensionality
reduction algorithms such as PCA, LLE, or MDS, and compare the resulting visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from chapter_08.common.timeit import timeit


def scatter_plot_reduction_results(results, algorithm_name, labels):
    plt.figure(figsize=(10, 8))  # Create a figure with specified size
    plt.scatter(
        results[:, 0],  # x-coordinates
        results[:, 1],  # y-coordinates,
        c=labels,  # color points by digit class,
        cmap="tab10",  # colormap with 10 distinct colors (perfect for digits 0-9)
        alpha=0.8,  # slight transparency
        s=10,  # point size
    )
    plt.colorbar(label="Digit")
    plt.title(f"{algorithm_name} visualization of MNIST digits")
    plt.xlabel(f"{algorithm_name} feature 1")
    plt.ylabel(f"{algorithm_name} feature 2")
    plt.savefig(
        f"chapter_08/ex10/{algorithm_name}_mnist_visualization.png"
    )  # save the image locally


@timeit
def run_single_dimensionality_reducer(X_train, dr_algorithm):
    result = dr_algorithm.fit_transform(X_train)
    return result


def main():
    # Step 1. Load and obtain reduced dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data
    y = mnist.target
    X_5000 = X[:5000]
    y_5000 = y[:5000]

    # Step 2. Run tSNE
    tsne_result, timing = run_single_dimensionality_reducer(
        X_5000, TSNE(random_state=42)
    )
    print(f"tSNE took {timing:.4f} seconds to execute.")

    # Convert the string labels to integers
    labels = y_5000.astype(int)

    # Step 3. Plot tSNE results using Matplotlib
    scatter_plot_reduction_results(tsne_result, "t-SNE", labels)

    # Step 4. Plot other dimensionality reduction algorithms
    dimensionality_reducing_algorithms = [
        ("PCA", PCA(n_components=2, random_state=42)),
        ("LLE", LocallyLinearEmbedding(n_components=2, random_state=42)),
        ("MDS", MDS(n_components=2, random_state=42, normalized_stress="auto")),
    ]
    for name, algorithm in dimensionality_reducing_algorithms:
        result, timing = run_single_dimensionality_reducer(X_5000, algorithm)
        print(f"{name} took {timing:.4f} seconds to execute.")
        scatter_plot_reduction_results(result, name, labels)


if __name__ == "__main__":
    main()
"""
The program prints:

    tSNE took 3.8849 seconds to execute.
    PCA took 0.0436 seconds to execute.
    LLE took 1.3875 seconds to execute.
    MDS took 391.1186 seconds to execute.
"""
