#POLYNOMIAL REGRESSION MODEL FOR A Y=Y(X) FUNCTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data reading

data = pd.read_csv('/Users/alessandropajer/Desktop/PHYSIS/22:23/VENTILATION/Static pressure plots/Dp_12vdc_100PWM.csv')
data

x = data['x']   #column name depends on your dataset
y = data[' y']

# Data visualization

plt.scatter(x, y, label = 'Distribution')
plt.xlabel('airflow [m^3/min]')
plt.ylabel('static pressure [Pa]')

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
plt.title('Polynomial regression pf fan ststic pressure @ 12VDC & 100% PMW', size = 16)
plt.scatter(x, y)
plt.plot(x, y, label = 'Distribution', color='navy')
plt.plot(x, y_predicted, label='Polynomial regression', color='orange', linewidth=4)
plt.xlabel('airflow [m^3/min]')
plt.ylabel('static pressure [Pa]')
plt.legend()
plt.show()
print('Intercept = ', poly_reg_model.intercept_) #Coeff. b0
print('Coefficents = ', poly_reg_model.coef_)  #Coeff. b1, b2, b3 ecc
print('MSE =', mean_squared_error(y, y_predicted)) #Mean squared error




#MORE DETAILED FUNCTION FOR DATA SPLIT IN TRAIN & TEST

"""m_poly = poly.fit_transform(m.values.reshape(-1, 1))

# Data split

m_train, m_test, p_train, p_test = train_test_split(poly_features, p, test_size = 0.10, random_state = 42)

p_predicted = model.predict(m_test)
p_predicted

# Control

MSE= mean_squared_error(p_test, p_pred)
print('MSE = {}'.format(MSE))

# Model visualization

p_pred_all = model.predict(m_poly)

plt.scatter(m, p, label = 'Distribution', color='navy')
plt.plot(m, p_pred_all,label='Polynomial regression', color='orange', linewidth = 4)
plt.xlabel('airflow [m^3/min]')
plt.ylabel('static pressure [Pa]')
plt.legend()
plt.show()"""





