# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:14:12 2022

@author: User
"""

#Method 1: Matrix multiplication

#Importing libraries
import os
#import csv
import numpy as np
import matplotlib.pyplot as plt

#Knowing current Python directory
current_directory = os.getcwd()
print(current_directory)

#Going to the working directory
working_directory = os.chdir(r'C:\Users\User\Desktop\Jose\Personal projects\Blog\Machine Learning\Linear Regression')
print(working_directory)

#Importing text file
data = np.loadtxt('points.txt', skiprows=(2), dtype=float)
print(data)

#Setting x values
x = data[:,0]
dimensions_x = x.shape
print(x)
print(dimensions_x)

#Setting y values
y = data[:,1]
dimensions_y = y.shape
print(y)
print(dimensions_y)

#Plotting data
plt.plot(x,y,'o')
plt.title('Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""
Linear Regression has the form y = mx + c or y = c0*x^0 + c1*x^1

x will be defined as a Vandermonde matrix

In linear algebra, a Vandermonde matrix is a matrix with terms of a geometric
progression in each row

V = [x1^0  x1^1  x1^2  ...]

Since we are interested in a linear relationship, the Vandermonde matrix:
V = [1  x1]
    [1  x2]
    [1  x3]
    .
    .
    .
    [1  xn]
    
Note: xn = len(x)

A.X = Y
A^(-1).A.X = A^(-1).Y
X = A^(-1).Y

The goal is to minimize the mean square error of the system. For this,
the coefficients will be defined as a matrix equal to:
    
A = (X.T*X)^(-1)*X.T*Y
"""


#Defining Vandermonde matrix
v = np.vstack((np.ones(len(x)),x)).T
print(v)
#Checking dimensions
dimensions_v = v.shape
print(dimensions_v)

#Defining the coefficient matrix
coeff = np.linalg.inv(v.T.dot(v)).dot(v.T).dot(y)
print(coeff)

y_lineal = v.dot(coeff)
print(y_lineal)

plt.scatter(x,y)
plt.plot(x, y_lineal, color='red')
plt.title('Matrix multiplication')
plt.xlabel('x')
plt.ylabel('y')
plt.show()