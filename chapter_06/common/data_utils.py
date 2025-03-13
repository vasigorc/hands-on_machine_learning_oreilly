from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def generate_moons_dataset(n_samples=10000, noise=0.4, random_state=42):
    """Generate the moons dataset with specified parameters."""
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def split_train_test(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
