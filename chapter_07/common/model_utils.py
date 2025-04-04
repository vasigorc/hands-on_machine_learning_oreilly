def evaluate_single_classifier(X_train, y_train, X_validation, y_validation, clf):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_validation, y_validation)
    return clf, accuracy


def train_and_evaluate_classifiers(
    classifiers, X_train, y_train, X_validation, y_validation
):
    """
    Train each classifier on the full training set and print its accuracy.

    Returns:
        trained_models: List of tuples (name, trained_classifier)
    """
    trained_models = []

    for name, clf in classifiers:
        fitted_clf, accuracy = evaluate_single_classifier(
            X_train, y_train, X_validation, y_validation, clf
        )
        print(f"{name}'s accuracy on the validation set is {accuracy:.4f}")

        trained_models.append((name, fitted_clf))

    return trained_models
