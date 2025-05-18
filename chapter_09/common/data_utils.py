from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split


def load_split_olivetti_dataset():
    olivetti = fetch_olivetti_faces()
    X_temp, X_test, y_temp, y_test = train_test_split(
        olivetti.data,
        olivetti.target,
        stratify=olivetti.target,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.2, random_state=42
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test
