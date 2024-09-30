import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .common.data_loader import get_train_test_data

# Constants
AGE = "Age"
SIBSP = "SibSp"
PARCH = "Parch"
FARE = "Fare"
PCLASS = "Pclass"
SEX = "Sex"
EMBARKED = "Embarked"
CABIN = "Cabin"
SURVIVED = "Survived"
FAMILY_SIZE = "FamilySize"
IS_ALONE = "IsAlone"
FARE_LOG = "Fare_log"
AGE_BAND = "AgeBand"
DECK = "Deck"


def print_evaluation_metrics(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} Evaluation Results:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision:\n{precision:.4f}")
    print(f"Recall:\n{recall:.4f}")
    print(f"F1-score:\n{f1:.4f}")


def evaluate_single_classifier(X_train, y_train, clf, name):
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5)

    print_evaluation_metrics(y_train, y_pred, name)


def evaluate_classifiers(X_train, y):
    classifiers = {
        "Random Forrest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVC": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
    }

    for name, clf in classifiers.items():
        evaluate_single_classifier(X_train, y, clf, name)


"""
Prints:

Classifier Evaluation results:

Random Forrest:
Confusion Matrix:
[[465  84]
 [102 240]]
Precision:
0.7407407407407407
Recall:
0.7017543859649122
F1-score:
0.7207207207207207

Logistic Regression:
Confusion Matrix:
[[462  87]
 [ 94 248]]
Precision:
0.7402985074626866
Recall:
0.7251461988304093
F1-score:
0.7326440177252586

SVC:
Confusion Matrix:
[[492  57]
 [ 99 243]]
Precision:
0.81
Recall:
0.7105263157894737
F1-score:
0.7570093457943925

KNN:
Confusion Matrix:
[[479  70]
 [109 233]]
Precision:
0.768976897689769
Recall:
0.6812865497076024
F1-score:
0.7224806201550388
"""


def tune_svc(X_train, y_train):
    param_distributions = {
        "C": np.logspace(
            -2, 2, 5
        ),  # regularization parameter, controls the trade-off bw achieving low training and testing errors (for unseen data). Smaller values - stronger regularization
        "kernel": [
            "rbf",
            "linear",
        ],  # the way the algorithm finds the hyperplane that separates the classes
        "gamma": [
            "scale",
            "auto",
            0.01,
            0.1,
            1,
        ],  # how far the influence of a single training example reaches. low values -> 'far', high values -> 'close'
    }

    svc = SVC(random_state=42)
    # Create the randomized search cv object to find best combination of the above parameter
    random_search = RandomizedSearchCV(
        svc,
        param_distributions,
        n_iter=10,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    print("Best SVC parameters found: ", random_search.best_params_)
    print(
        "Best SVC cross-validation F1 score: {:.4f}".format(random_search.best_score_)
    )
    evaluate_single_classifier(X_train, y_train, svc, "Final model")
    """
    For now this prints:
    Final model Evaluation Results:
    Confusion Matrix:
    [[492  57]
    [ 99 243]]
    Precision:
    0.8100
    Recall:
    0.7105
    F1-score:
    0.7570
    """

    return random_search.best_estimator_


def train_final_model(X_train, y_train, best_model):
    best_model.fit(X_train, y_train)
    best_model


def add_family_size_and_is_alone(X):
    family_size = (X[:, 0] + X[:, 1] + 1).reshape(-1, 1)
    is_alone = (family_size == 1).astype(int).reshape(-1, 1)
    return np.hstack([family_size, is_alone])


def family_size_is_alone_name(function_transformer, feature_names_in):
    return [FAMILY_SIZE, IS_ALONE]


def family():
    return make_pipeline(
        KNNImputer(n_neighbors=3),
        FunctionTransformer(
            add_family_size_and_is_alone, feature_names_out=family_size_is_alone_name
        ),
    )


def log_transform_fare(X):
    return np.log1p(X)


def log_fare_name(function_transformer, feature_names_in):
    return [FARE_LOG]


def fill_embarked(X):
    return X.fillna("U")


def embarked_name(function_transformer, feature_names_in):
    return [EMBARKED]


def extract_deck(X):
    return (
        pd.DataFrame(X)
        .fillna("Unknown")
        .astype(str)
        .iloc[:, 0]
        .str[0]
        .replace("", "U")
        .values.reshape(-1, 1)
    )


def deck_name(function_transformer, feature_names_in):
    return [DECK]


def create_age_bands(X):
    return pd.cut(
        X.iloc[:, 0],
        bins=[0, 12, 18, 65, np.inf],
        labels=["Child", "Teenager", "Adult", "Senior"],
    ).values.reshape(-1, 1)


def age_band_name(function_transformer, feature_names_in):
    return [AGE_BAND]


def create_preprocessor():
    return ColumnTransformer(
        [
            ("family", family(), [SIBSP, PARCH]),
            (
                "fare",
                make_pipeline(
                    KNNImputer(n_neighbors=3),
                    FunctionTransformer(
                        log_transform_fare, feature_names_out=log_fare_name
                    ),
                    StandardScaler(),
                ),
                [FARE],
            ),
            (
                "pclass",
                make_pipeline(KNNImputer(n_neighbors=3), StandardScaler()),
                [PCLASS],
            ),
            ("sex", OneHotEncoder(sparse_output=False), [SEX]),
            (
                "embarked",
                make_pipeline(
                    FunctionTransformer(fill_embarked, feature_names_out=embarked_name),
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                ),
                [EMBARKED],
            ),
            (
                "deck",
                Pipeline(
                    [
                        (
                            "extract_deck",
                            FunctionTransformer(
                                extract_deck, feature_names_out=deck_name
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                        ),
                    ]
                ),
                [CABIN],
            ),
            (
                "age_band",
                Pipeline(
                    [
                        (
                            "create_bands",
                            FunctionTransformer(
                                create_age_bands, feature_names_out=age_band_name
                            ),
                        ),
                        ("onehot", OneHotEncoder(sparse_output=False)),
                    ]
                ),
                [AGE],
            ),
        ]
    )


def main():
    train_set, test_set = get_train_test_data()

    # Create preprocessor
    preprocessor = create_preprocessor()

    # Fit and process the data
    X_processed = preprocessor.fit_transform(train_set)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create a DataFrame with the processed data
    processed_df = pd.DataFrame(
        X_processed, columns=feature_names, index=train_set.index
    )

    # Add the target value
    processed_df[SURVIVED] = train_set[SURVIVED]

    # Calculate correlations
    correlation_matrix = processed_df.corr()
    survival_correlations = (
        correlation_matrix[SURVIVED].drop(SURVIVED).sort_values(ascending=False)
    )

    print(f"Correlations with {SURVIVED} (excluding {SURVIVED} itself):")
    print(survival_correlations)

    # Evaluate classifiers
    X_train = processed_df.drop(SURVIVED, axis=1)
    y_train = processed_df[SURVIVED]

    evaluate_classifiers(X_train, y_train)

    print("\nTuning SVC (best model)...")
    best_svc = tune_svc(X_train, y_train)

    # train final model
    final_model = train_final_model(X_train, y_train, best_svc)

    # process test set
    # Note: test_set purposefully doesn't contain SURVIVED model
    X_test_processed = preprocessor.transform(test_set)
    X_test = pd.DataFrame(X_test_processed, columns=feature_names, index=test_set.index)


if __name__ == "__main__":
    main()
