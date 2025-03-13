from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def find_best_hyperparameters(X_train, y_train, random_state=42):
    """
    Find the best hyperparameters for a DecisionTreeClassifier using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility

    Returns:
        grid_search: Fitted GridSearchCV object with best parameters
    """
    tree_clf = DecisionTreeClassifier(random_state=random_state)
    param_grid = [
        {"max_depth": [None, 3, 6, 9, 12, 14]},
        {"max_leaf_nodes": [10, 20, 30, 40, 50, 80, 100]},
    ]

    grid_search = GridSearchCV(
        tree_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    return grid_search


def train_model_with_best_params(X_train, y_train, best_params, random_state=42):
    """
    Train a new model with the best parameters found.

    Args:
        X_train: Training features
        y_train: Training labels
        best_params: Dictionary of best parameters
        random_state: Random seed for reproducibility

    Returns:
        Trained DecisionTreeClassifier
    """
    best_tree = DecisionTreeClassifier(random_state=random_state, **best_params)
    best_tree.fit(X_train, y_train)
    return best_tree


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data."""
    accuracy = model.score(X_test, y_test)
    return accuracy
