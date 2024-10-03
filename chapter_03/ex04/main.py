from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from .data_loader import DatasetName, load_emails
from .preprocessing import preprocessor_pipeline
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split


def main():
    # 1. load data
    spam_emails = load_emails(DatasetName.SPAM)
    ham_emails = load_emails(DatasetName.HAM)

    # 2. split into train test and data
    X = np.array(ham_emails + spam_emails, dtype=object)
    # set labels according to subset source
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. preprocess the training dataset
    X_train_transformed = preprocessor_pipeline.fit_transform(X_train)
    # 4. Evaluate score for LogisticRegression model
    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3)
    print(f"Mean score for LogisticRegression is {score.mean()}")
    # 5. Print out the precision and recall for the test set
    X_test_transformed = preprocessor_pipeline.transform(X_test)

    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    log_clf.fit(X_test_transformed, y_test)

    y_pred = log_clf.predict(X_test_transformed)

    print(f"Precision: {precision_score(y_test, y_pred): .2%}")
    print(f"Recall: {recall_score(y_test, y_pred): .2%}")
    """
    The program is expected to print:
        python3 -m chapter_03.ex04.main
        Mean score for LogisticRegression is 0.985
        Precision:  100.00%
        Recall:  100.00%
    """


if __name__ == "__main__":
    main()
