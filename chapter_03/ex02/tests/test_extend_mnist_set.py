import pytest
import numpy as np
from ex02.extend_mnist_set import get_image_shifts, create_expanded_dataset


@pytest.fixture
def test_image():
    # Create a simple 3x3 test image
    return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


def test_output_length(test_image):
    shifts = get_image_shifts(test_image)
    assert len(shifts) == 4, "Should return 4 shifted images"


def test_left_shift(test_image):
    shifts = get_image_shifts(test_image)
    expected_left_shift = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(shifts[0], expected_left_shift)


def test_right_shift(test_image):
    shifts = get_image_shifts(test_image)
    expected_right_shift = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(shifts[1], expected_right_shift)


def test_up_shift(test_image):
    shifts = get_image_shifts(test_image)
    expected_up_shift = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(shifts[2], expected_up_shift)


def test_down_shift(test_image):
    shifts = get_image_shifts(test_image)
    expected_down_shift = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(shifts[3], expected_down_shift)


def test_with_real_image():
    # Test with a more complex image, e.g., a small MNIST-like image
    test_image = np.random.rand(28, 28)  # Random 28x28 image
    shifts = get_image_shifts(test_image)

    assert len(shifts) == 4, "Should return 4 shifted images"
    assert all(
        shift.shape == test_image.shape for shift in shifts
    ), "All shifts should have the same shape as the input"


def test_with_empty_image():
    empty_image = np.array([])
    with pytest.raises(ValueError, match="Input image is empty"):
        get_image_shifts(empty_image)


def test_create_expanded_dataset():
    X = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])  # One 3x3 image flattened
    y = np.array([5])

    X_expanded, y_expanded = create_expanded_dataset(X, y, expected_shape=(3, 3))

    assert X_expanded.shape == (5, 9), "Should have 5 images (1 original + 4 shifts)"
    assert y_expanded.shape == (5,), "Should have 5 labels"
    assert np.all(y_expanded == 5), "All labels should be the same as the original"


def test_create_expanded_dataset_empty():
    with pytest.raises(ValueError):
        create_expanded_dataset(np.array([]), np.array([]))
