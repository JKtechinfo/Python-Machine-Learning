# -*- coding: utf-8 -*-
"""
Created on Sun May  8 00:40:39 2022

@author: User
"""


#Importing libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Knowing current Python directory
current_directory = os.getcwd()
print(current_directory)

#Going to the working directory
working_directory = os.chdir(r'C:\Users\User\Desktop\Jose\Personal projects\Blog\Machine Learning\Linear Regression\Matrix multiplication')
print(working_directory)

#Importing text file
data = np.loadtxt('points.txt', skiprows=(2), dtype=float)
print(data)

#Setting x values
x = data[:,0]
print(x)

#Setting y values
y = data[:,1]
print(y)

#Plotting data
plt.plot(x,y,'o')
plt.title('Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Setting the polynomial degree
degree = PolynomialFeatures(degree = 5)

#Converting the 1D-array into 2D-array
x_reshape = x.reshape(-1,1)
print(x_reshape)

#Fitting the polynomial to the degree set previously
polynomial = degree.fit(x_reshape)

#Creating our Vandermonde matrix according to the degree set previously
Vandermonde_matrix = degree.transform(x_reshape)

#Creating the polynomial regression model
model = LinearRegression()

#Training the model (Fitting the Vandermonde matrix to the y-values)
train_model = model.fit(Vandermonde_matrix, y)

#Getting the predicted y-values according to our x-Vandermonde matrix
y_predicted = train_model.predict(Vandermonde_matrix)

#Plotting the Polynomial Regression model
plt.plot(x,y,'o')
plt.plot(x,y_predicted,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()