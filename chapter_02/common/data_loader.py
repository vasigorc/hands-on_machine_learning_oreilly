import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)

DATASET_URL = "https://github.com/ageron/data/raw/main/housing.tgz"
DATASET_DIR = Path(__file__).parent.parent / "datasets"
TARBALL_PATH = DATASET_DIR / "housing.tgz"
CSV_PATH = DATASET_DIR / "housing/housing.csv"


def download_and_extract_housing_data():
    if not TARBALL_PATH.is_file():
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, TARBALL_PATH)
        with tarfile.open(TARBALL_PATH) as housing_tarball:
            housing_tarball.extractall(path=DATASET_DIR)


def load_housing_data():
    if not CSV_PATH.is_file():
        download_and_extract_housing_data()
    return pd.read_csv(CSV_PATH)


def get_train_test_data():
    housing = load_housing_data()
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing, housing_labels, strat_test_set
