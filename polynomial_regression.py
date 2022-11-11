#POLYNOMIAL REGRESSION MODEL FOR A Y=Y(X) FUNCTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data reading

data = pd.read_csv('yourFile.csv')
data

x = data['x']   #column name depends on your dataset
y = data[' y']

# Data visualization

plt.scatter(x, y, label = 'Distribution')
plt.xlabel('x axis')
plt.ylabel('y axis')

plt.show()

poly = PolynomialFeatures(degree = 6, include_bias = False)

poly_features = poly.fit_transform(x.values.reshape(-1, 1))  # With fit() we basically just declare what feature we want to transform, transform() performs the actual transformation

# Regression model

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)

y_predicted = poly_reg_model.predict(poly_features)
y_predicted # Predicted responses

# Model visualization

plt.figure(figsize= (10, 6))
plt.title('Polynomial regression', size = 16)
plt.scatter(x, y)
plt.plot(x, y, label = 'Distribution', color='navy')
plt.plot(x, y_predicted, label='Polynomial regression', color='orange', linewidth=4)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.show()
print('Intercept = ', poly_reg_model.intercept_) #Coeff. b0
print('Coefficents = ', poly_reg_model.coef_)  #Coeff. b1, b2, b3 ecc
print('MSE =', mean_squared_error(y, y_predicted)) #Mean squared error





