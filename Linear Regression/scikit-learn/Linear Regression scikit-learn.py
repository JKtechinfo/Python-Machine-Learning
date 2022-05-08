# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 18:37:57 2022

@author: User
"""


#Importing libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Importing text file
data = np.loadtxt('points.txt', skiprows=(2), dtype=float)
print(data)

#Setting x values
x = data[:,0]
print(x)

#Getting the dimensions of x
dimensions_x = x.shape
print(dimensions_x)

#Setting y values
y = data[:,1]
print(y)

#Getting the dimensions of y
dimensions_y = y.shape
print(dimensions_y)

#Reshaping the array into a vector-column
x2 = data[:,0].reshape(-1,1)
print(x2)

#Getting the dimensions of x2
dimensions_x2 = x2.shape
print(dimensions_x2)

#Reshaping the array into a vector-column
y2 = data[:,1].reshape(-1,1)
print(y2)

#Getting the dimensions of y2
dimensions_y2 = y2.shape
print(dimensions_y2)

#Building the linear regression model
linear_regression = LinearRegression()
linear_model = linear_regression.fit(x2,y2)

#Getting the intercept with y-axis
intercept_yaxis = linear_model.intercept_
print(intercept_yaxis)

#Getting the coefficient
slope = linear_model.coef_
print(slope)

#Establishing the linear relationship
y_lineal2 = slope*x2 + intercept_yaxis
print(y_lineal2)

#Plotting

#Initially given x- and y-points
plt.scatter(x,y)
#Linear regression points
plt.plot(x2, y_lineal2, color='red')
#Naming the graph, x- and y-axis
plt.title('scikit-learn library')
plt.xlabel('x')
plt.ylabel('y')
plt.show()