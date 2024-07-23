import pandas as pd
import pytest
import numpy as np
from sklearn.utils import check_random_state
from ex06.main import StandardScalerClone


@pytest.fixture
def random_data():
    rng = check_random_state(42)
    return rng.rand(1000, 3)


def test_standard_scaler_clone(random_data):
    scaler = StandardScalerClone()
    X_scaled = scaler.fit_transform(random_data)

    # Manual scaling
    X_manual_scaled = (random_data - random_data.mean(axis=0)) / random_data.std(axis=0)

    # allclose returns True if two arrays are element-wise equal within a tolerance
    assert np.allclose(X_scaled, X_manual_scaled)


def test_with_means_flag(random_data):
    scaler = StandardScalerClone(with_mean=False)
    X_scaled_uncentered = scaler.fit_transform(random_data)

    assert np.allclose(X_scaled_uncentered, random_data / random_data.std(axis=0))


def test_inverse_transform(random_data):
    scaler = StandardScalerClone()
    X_scaled = scaler.fit_transform(random_data)
    X_inverse = scaler.inverse_transform(X_scaled)

    # Check if the inverse transform returns close to original data
    assert np.allclose(random_data, X_inverse)


def test_feature_names_out():
    scaler = StandardScalerClone()
    X = np.random.rand(100, 3)
    scaler.fit(X)

    # Test with no input features
    assert np.array_equal(scaler.get_feature_names_out(), np.array(["x0", "x1", "x2"]))

    # Test with input features
    input_features = ["a", "b", "c"]
    assert np.array_equal(
        scaler.get_feature_names_out(input_features), np.array(input_features)
    )


def test_invalid_feature_names():
    scaler = StandardScalerClone()
    X = np.random.rand(100, 3)
    scaler.fit(X)

    with pytest.raises(ValueError):
        scaler.get_feature_names_out(["a", "b"])  # Wrong number of features
