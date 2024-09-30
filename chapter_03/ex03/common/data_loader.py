from data_utilities.data_loader import DataLoader

DATASET_URL = "https://homl.info/titanic.tgz"
DATASET_NAME = "titanic"

titanic_data_loader = DataLoader(DATASET_NAME, DATASET_URL)

def get_train_test_data():
  train_set = titanic_data_loader.load_data("train.csv")
  test_set = titanic_data_loader.load_data("test.csv")
  return train_set, test_set
