from sklearn.metrics import accuracy_score

from common import (
    X_train,
    X_test,
    y_train,
    y_test,
    fit_and_get_best_model,
)

random_search = fit_and_get_best_model(X_train, y_train)

best_model = random_search.best_estimator_
# get the best parameters
# this should equal `{'n_neighbors': 7, 'weights': 'distance'}` and will be used by `random_search` for prediction
best_params = random_search.best_params_


# predict on the training data with the best model
y_pred = best_model.predict(X_test)

# calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"KNeighborsClassifier's regularized accuracy is {accuracy * 100:.2f}%")
