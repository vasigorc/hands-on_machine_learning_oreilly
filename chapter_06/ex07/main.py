# chapter_06/ex07/main.py
from chapter_06.common.data_utils import generate_moons_dataset, split_train_test
from chapter_06.common.model_utils import (
    evaluate_model,
    find_best_hyperparameters,
    train_model_with_best_params,
)


def main():
    # Step a: Generate the moons dataset
    X, y = generate_moons_dataset()

    # Step b: Split into training and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step c: Find best hyperparameters
    grid_search = find_best_hyperparameters(X_train, y_train)

    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test score with grid search model: {grid_search.score(X_test, y_test):.4f}")

    # Step d: Train on full training set with best parameters
    best_model = train_model_with_best_params(
        X_train, y_train, grid_search.best_params_
    )

    # Evaluate on test set
    test_accuracy = evaluate_model(best_model, X_test, y_test)
    print(f"Final model's test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
