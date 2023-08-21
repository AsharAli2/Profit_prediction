# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

#  getting info of our dataset
#  print(dataset.info())
#  0   R&D Spend        50 non-null     float64
#  1   Administration   50 non-null     float64
#  2   Marketing Spend  50 non-null     float64
#  3   State            50 non-null     object
#  4   Profit           50 non-null     float64

X = dataset[["R&D Spend","Administration","Marketing Spend"]]

y = dataset[["Profit"]]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

print(score)
# score: 0.9393955917820571 Very good accuracy
