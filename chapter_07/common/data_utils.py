from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist_dataset():
    """Load the MNIST dataset"""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data
    y = mnist.target
    return X, y


def split_mnist_dataset(
    X, y, validation_size=10_000, test_size=10_000, random_state=42
):
    """
    Split MNIST dataset into training, validation, and test sets.

    Args:
        X: Features (MNIST images)
        y: Labels (digits 0-9)
        validation_size: Number of samples for training set (default: 10_000)
        test_size: Number of samples for test set (default: 10_000)
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_validation, X_test, y_train, y_validation, y_test: Split datasets

    Note:
        The validation set will contain all remaining samples not in train or test sets.
    """
    # First split to get the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # split the remaining data to get training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp, y_temp, test_size=validation_size, random_state=random_state
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_validation.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_validation, X_test, y_train, y_validation, y_test
