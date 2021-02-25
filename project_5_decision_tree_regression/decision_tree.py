#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: theepag
"""

# Step 1 - Load Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Step 2 - Fit Decision Tree Regressor
regressor = DecisionTreeRegressor(criterion="mse")
regressor.fit(X, y)

# Step 3 - Visualize

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Decision Tree Regressor")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# Step 4 - Predict
y_pred = regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level is ', y_pred)
