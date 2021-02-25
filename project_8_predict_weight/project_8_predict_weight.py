#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: theepag
"""

# Step 1 : Load Dataset
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Height_Weight_single_variable_data_101_series_1.0.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Step 2: Check for missing values
dataset.isnull().sum()

# Step 3: Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Step 4: Fit Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 5: Predict values for test data
lin_pred = lin_reg.predict(X_test)

# Step 6: Compare predictions with real results
print('R square = ', metrics.r2_score(y_test, lin_pred))
print('Mean squared Error = ', metrics.mean_squared_error(y_test, lin_pred))


# Step 7: Visualize Training set
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, lin_reg.predict(X_train), color="blue")
plt.title("Height and Weight - Training Set")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# Step 8: Visualize Test set
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, lin_reg.predict(X_train), color="blue")
plt.title("Height and Weight - Test Set")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# Step 9: Make new Prediction
lin_pred_new = lin_reg.predict([[166]])
print('If a person has height 166, the predicted weight is ', lin_pred_new)
