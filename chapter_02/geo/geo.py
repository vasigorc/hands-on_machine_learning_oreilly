from common import TransformerFromRegressor, get_preprocessor, get_train_test_data
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor

"""
Try creating a custom transformer that trains a _k_ - nearest neighbors regressor 
(`sklearn.neighbors.KNeighborsRegressor`) in its `fit()` method, and outputs the model's predictions 
in its `transform()` method. Then add this feature to the preprocessing pipeline, using latitude and 
longitude as the inputs to this transformer. This will add a feature in the model that corresponds 
to the housing median price of the nearest districts.
"""

knn_regressor = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_transformer = TransformerFromRegressor(knn_regressor)

# now we have to tweak our default preprocessing to include this transformer
preprocessing = get_preprocessor()
transformers = [
    (name, clone(transformer), columns)
    for name, transformer, columns in preprocessing.transformers
]
geo_index = [name for name, _, _ in transformers].index("geo")
transformers[geo_index] = ("geo", knn_transformer, ["latitude", "longitude"])
prepreocessing_with_geo = ColumnTransformer(transformers)
