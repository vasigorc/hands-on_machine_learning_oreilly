import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Download and prepare the data
data_root = "https://raw.githubusercontent.com/ageron/data/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x_label: str = "GDP per capita (USD)"
y_label: str = "Life satisfaction"
x = lifesat[[x_label]].values
y = lifesat[[y_label]].values

# Visualize the data
lifesat.plot(kind="scatter", grid=True, x=x_label, y=y_label)
plt.axis((23_500, 62_500, 4, 9))
plt.show()

# Select a k-neighbour regression model
model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(x, y)

# Make a prediction for Cyprus
x_new = [[37_652.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(x_new))
