
from pandas.io.formats.style_render import DataFrame
from sklearn.model_selection import train_test_split

from data_utilities.data_loader import DataLoader

DATASET_URL = "https://github.com/ageron/data/raw/main/housing.tgz"
DATASET_NAME = "housing"
CSV_FILENAME = f"{DATASET_NAME}.csv"

housing_loader = DataLoader(DATASET_NAME, DATASET_URL)

def get_train_test_data():
    housing = housing_loader.load_data(CSV_FILENAME)
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )
    strat_train_set = DataFrame(strat_train_set)
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing, housing_labels, strat_test_set
