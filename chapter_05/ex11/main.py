from sklearn.datasets import fetch_california_housing

"""
Exercise 11. Train and fine-tune an SVM regressor on the California housing dataset. You can use the
original dataset rather than the tweaked version we used in Chapter 2, which you can load using `sklearn.datasets.fetch_california_housing()`.
The targets represent hundreds of thousands of dollars. Since there are over 20000 instances, SVMs can
be slow, so for hyperparameters tuning you should use far fewer instances (e.g. 2000) to test many more
hyperparameter combinations. What is your best model's RMSE?
"""


def main():
    housing = fetch_california_housing()
    pass


if __name__ == "__main__":
    main()
