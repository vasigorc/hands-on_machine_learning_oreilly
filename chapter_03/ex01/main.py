from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

"""
`fetch_openml` returns the inputs as a Panda DataFrame and the labels as a Pandas Series. Since MNIST
dataset contains images, and DataFrames aren't ideal for that, so it's preferable to set as_frame=False
to get the data as NumPy arrays instead 
"""
mnist = fetch_openml("mnist_784", as_frame=False)

# tuple containing the input data and the targets
X, y = mnist.data, mnist.target

"""
MNIST dataset returned by `fetch_openml` is already split into a training set (first 60k images) and
a test set (last 10k images). Training set is already shuffled
"""
X_train, X_test, y_train, y_test = X[:60_000], X[60_000:], y[:60_000], y[60_000:]

y_train_5 = y_train == "5"  # True for all 5s

"""
Exercise 1. Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the 
test set. Hint: the `KNeighborsClassifier` works quite well for this task; you just need to find 
good hyperparameter values (try a grid search on the weights and n_neighbors hyperparameters)
"""
