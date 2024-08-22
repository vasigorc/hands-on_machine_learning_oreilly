from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift

"""
Write a function that can shift MNIST image in any direction (left, right, up, or down) by one pixel 
(you can use `shift()` function from `scipy.interpolation` module. For example, 
`shift(image, [2, 1], cval=0)` shifts the image two pixels down and one pixel to the right). 
Then, for each image in the training set, create four shifted copies (one per direction) and add them 
to the training set. Finally, train your best model on this expanded training set and measure its 
accuracy on the test set. You should observe that your model performs even better now! This technique 
of artificially growing the training set is called _data augemntation_ or _training set expansion_
"""

"""
takes a MNIST image represented as numpy array and returns 4 copies of it, each shifted into
different directions by 1 pixel
"""


def get_image_shifts(image: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    if image.size == 0:
        raise ValueError("Input image is empty")

    return [
        shift(image, [0, -1], cval=0),  # left
        shift(image, [0, 1], cval=0),  # right
        shift(image, [-1, 0], cval=0),  # up
        shift(image, [1, 0], cval=0),  # down
    ]


"""
takes as input:

X - a MNIST training set and
y - its labels
expected_shape - height/weight of each image

returns: initial array supplemented with copies of each image as specified by get_image_shifts function
"""


def create_expanded_dataset(
    X: NDArray[np.float64],
    y: NDArray[np.str_],
    expected_shape: Tuple[int, int] = (28, 28),
) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    if X.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be the same")

    X_expanded: list[NDArray[np.float64]] = []
    y_expanded: list[np.str_] = []

    for image, label in zip(X, y):
        image_2d: NDArray[np.float64] = image.reshape(expected_shape)

        # Add original image
        X_expanded.append(image)
        y_expanded.append(label)

        # Add shifted images
        shifted_images: list[NDArray[np.float64]] = get_image_shifts(image_2d)
        X_expanded.extend([img.flatten() for img in shifted_images])
        y_expanded.extend([label] * 4)

    return np.array(X_expanded), np.array(y_expanded)
