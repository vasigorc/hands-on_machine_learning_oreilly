from common.data_loader import X_train, y_train, X_test, y_test
from common.knn_clif_fine_tuned import fit_and_get_best_model
from extend_mnist_set import create_expanded_dataset
from sklearn.metrics import accuracy_score

# Create the expanded dataset\
X_train_expanded, y_train_expanded = create_expanded_dataset(X_train, y_train)

print(f"Original dataset shape: {X_train.shape}")
print(f"Expanded dataset shape: {X_train_expanded.shape}")

"""
Don't do this at home! With:

```
nproc --all
32
```

and

```
free -h
               total        used        free      shared  buff/cache   available
Mem:            62Gi        16Gi        27Gi       1.3Gi        18Gi        40Gi
Swap:           19Gi          0B        19Gi
```

this training took ~30 minutes
"""
ex2_random_search = fit_and_get_best_model(X_train_expanded, y_train_expanded)

# This should print: best params are {'n_neighbors': 8, 'weights': 'distance'}
print(f"best params are {ex2_random_search.best_params_}")

y_pred = ex2_random_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Prints: KNeighborsClassifier's regularized accuracy for augmented training set is 97.73%
print(
    f"KNeighborsClassifier's regularized accuracy for augmented training set is {accuracy * 100:.2f}%"
)
