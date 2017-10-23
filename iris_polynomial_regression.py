# This is an exmaple of classification on the iris dataset
# using polynomial regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# import statsmodels.formula.api as sm

iris = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)

# Assign names for the 5 columns if needed
# iris.columns = ['SEPAL LENTH','SEPAL WIDTH','PETAL LENTH','PETAL WIDTH','CLASS']

X = iris.iloc[:,:-1].values
y = iris.iloc[:,4].values

X = np.append(arr = np.ones((150,1)).astype(int), values = X, axis=1)

# Encode class
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split dataset into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting using multiple linear regression on train set
# All independent variable used
poly_reg = PolynomialFeatures(degree = 4)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
regressor = LinearRegression()
regressor.fit(X_poly_train, y_train)

# Test on the test set
y_pred = regressor.predict(X_poly_test)
plt.figure(1)
plt.hist(y_pred.round() - y_test)
