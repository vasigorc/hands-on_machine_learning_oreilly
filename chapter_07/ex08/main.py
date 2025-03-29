"""
Exercise 8:

Load the MNIST dataset (introduced in Chapter 3), and split it into a training set, a validation set,
and a test set (e.g. use 50*000 instances for training, 10_000 each for validation and testing). Then
train the various classifiers, such as a random forest classifier, an extra-treees classifier, and
a SVM classifier. Next, try to combine them into an ensemble that outperforms each individual classifier
on the validation set, using _soft_ or _hard_ voting. Once you have found one, try it on the test
set. How much better does it perform compared to the individual classifiers?
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def main():
    digits = load_digits(as_frame=True)
    n_samples = len(digits.images)
    print(f"# of samples: {n_samples}")


if __name__ == "__main__":
    main()
