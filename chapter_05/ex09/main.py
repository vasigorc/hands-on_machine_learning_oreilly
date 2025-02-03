import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier

"""
Exercise 9. Train a `LinearSVC` on a linearly separable dataset. Then train an `SVC` and a `SGDClassifier`
on the same dataset. See if you can get them to produce roughly the same model.
"""


def main():
    # Generate a linearly separable dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,  # Two features for easy visualization
        n_informative=2,  # Both features will contribute to the separation
        n_redundant=0,  # No redundant features
        n_clusters_per_class=1,  # One cluster per class
        class_sep=2.0,  # Increase class separation for better linear separability
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # training LinearSVC
    linear_svc_model = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
    linear_svc_model.fit(X_train, y_train)
    plot_decision_boundary(
        X_train, y_train, linear_svc_model, title="LinearSVC Decision Boundary"
    )

    # training SVC with linear kernel
    svc_model = make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=42))
    svc_model.fit(X_train, y_train)
    plot_decision_boundary(X_train, y_train, svc_model, title="SVC Decision Boundary")

    # training SGDClassifier
    sgd_model = make_pipeline(
        StandardScaler(), SGDClassifier(loss="hinge", alpha=0.001, random_state=42)
    )
    sgd_model.fit(X_train, y_train)
    plot_decision_boundary(
        X_train, y_train, sgd_model, title="SGDClassifier Decision Boundary"
    )


def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    # Create a mesh grid to visualize the decision boundary
    #  X[:, 0] -  means first column of all rows
    #  `-1` and `+1` are used as "breathing room", an artificially added margin in order to be able to visualize all data points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    """
        Mesh grid works as follows:

        x_array = np.array([1, 2, 3])
        y_array = np.array([4, 5])

        xx, yy = np.meshgrid(x_array, y_array)

        xx becomes:
        [[1, 2, 3],  # First row (pairs with y=4)
        [1, 2, 3]]   # Second row (pairs with y=5)

        yy becomes:
        [[4, 4, 4],   # First row (pairs with x=1,2,3)
        [5, 5, 5]]    # Second row (pairs with x=1,2,3)
    """
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Transform the grid into a list of points
    # xx.ravel() flattens the grid into a 1D array
    # np.c_[] concatenates arrays column-wise
    # creates pairs of (x, y) coordinates for prediction
    Z = model.predict(mesh_points)
    # reshapes predictions back to match the grid shape
    Z = Z.reshape(xx.shape)

    # Get the decision function values (distance from the hyperplane)
    # Note: we need to use the named step from the pipeline
    final_estimator = model._final_estimator
    Z_scores = None
    if hasattr(final_estimator, "decision_function"):
        # Plot margins
        Z_scores = final_estimator.decision_function(mesh_points)
        Z_scores = Z_scores.reshape(xx.shape)

    # create the plot with size 10x8 inches
    plt.figure(figsize=(10, 8))

    # plot the decision boundary, by creating contours (decision regions), and make them semi-transparent
    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, alpha=0.4)

    if Z_scores is not None:
        # Draw the decision boundary
        plt.contour(xx, yy, Z_scores, levels=[0], colors="k", linestyles="-")
        # Draw the margins
        plt.contour(xx, yy, Z_scores, levels=[-1, 1], colors="k", linestyles="--")

    # plot the actual data points (x, and then y coordinates) and make them slightly transparent
    # `c=y` colors points based on their class labels
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.colorbar(scatter)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)

    # Save the plot instead of displaying in
    plt.savefig(f"chapter_05/ex09/{title}.png")
    plt.close()  # close the figure to free memory


if __name__ == "__main__":
    main()
