from sklearn.datasets import fetch_openml

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
