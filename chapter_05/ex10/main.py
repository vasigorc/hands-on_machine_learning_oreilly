from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

"""
Exercise 10. Train an SVM classifier on the wine dataset, which you can load using `sklearn.datasets.load_wine`.
This dataset contains the chemical analyses of 178 wine samples produced by 3 different cultivators:
the goal is to train a classification model capable of predicting the cultivator based on the wine's
chemical analysis. Since SVM classifiers are binary classifiers, you will need to use one-versus-all
to classify all three classifiers. What accuracy can you reach?

Please see the Jupyter Notebook of this exercise for detailed steps, and analysis.
"""


def main():
    wine = load_wine(as_frame=True)

    X = wine.data
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # find optimal hyperparameters
    # During investigation we've defined that the dataset has linear nature, hence usage of LinearSVC
    model_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("svc", LinearSVC(random_state=42, dual=False))]
    )
    param_grid = [
        {"svc__C": [0.01, 0.1, 1, 10, 100], "svc__max_iter": [1000, 2000, 3000]}
    ]
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    print(
        "Best parameters: ", grid_search.best_params_
    )  # should be {'svc__C': 0.01, 'svc__max_iter': 1000}
    print("Best score: ", grid_search.best_score_)  # 0.9788177339901478

    # validate the model
    best_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=0.01, max_iter=1000, dual=False, random_state=42)),
        ]
    )

    # Get the predictions using cross-validation
    y_pred = cross_val_predict(best_model, X_train, y_train, cv=5)

    # fit the optimal model and evaluate it with the test set
    best_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = best_model.predict(X_test)

    # Evaluate
    print("Final result:")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report: ")
    print(classification_report(y_test, y_pred))

    """
        Confusion Matrix: 
        [[14  0  0]
        [ 0 14  0]
        [ 0  0  8]]

Classification Report: 
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        14
           1       1.00      1.00      1.00        14
           2       1.00      1.00      1.00         8

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
    """


if __name__ == "__main__":
    main()
