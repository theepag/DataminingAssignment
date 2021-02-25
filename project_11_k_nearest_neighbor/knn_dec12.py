#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: theepag
"""

# Step 1 - Load Data
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
dataset = pd.read_csv("iphone_purchase_records.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Step 2 - Convert Gender to number
labelEncoder_gender = LabelEncoder()
X[:, 0] = labelEncoder_gender.fit_transform(X[:, 0])

# Optional - if you want to convert X to float data type
X = np.vstack(X[:, :]).astype(np.float)


# Step 3 - Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)


# Step 4 - Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Step 5 - Fit KNN Classifier
# metric = minkowski and p=2 is Euclidean Distance
# metric = minkowski and p=1 is Manhattan Distance
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)

# Step 5 - Make Prediction
y_pred = classifier.predict(X_test)

# Step 6 - Confusion Matrix
#from sklearn import metrics
# cm = metrics.confusion_matrix(y_test, y_pred) ## 4,3 errors
# accuracy = metrics.accuracy_score(y_test, y_pred) ## 0.93
# precision = metrics.precision_score(y_test, y_pred) ## 0.87
# recall = metrics.recall_score(y_test, y_pred) ## 0.90

# Step 7 - Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)
precision = metrics.precision_score(y_test, y_pred)
print("Precision score:", precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall score:", recall)