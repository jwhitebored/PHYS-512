# -*- coding: utf-8 -*-
"""
PHYS 512
Assignment 1 Problem 1
@author: James Nathan White (260772425)
"""
#I always import this stuff
import matplotlib.pyplot as plt
import random as r
import glob
import numpy as np
from scipy.stats import chi2
import scipy.optimize as opt
from scipy.stats import norm
import pandas as pd

#The test values of x
x = np.linspace(0, 5, 11)

#The Machine Epsilon Values
single_epsilon = np.power(10.0, -8)
double_epsilon = np.power(10.0, -16)

#Optimal delta values (see attached pdf for problem 1)
optimal_single_delta = 2.5*np.power(10.0, -2)
optimal_double_delta = 6.3*np.power(10.0, -4)

#List of different delta values to see which is roughly best
double_deltas = np.zeros(16)
for i in range(len(double_deltas)):
    double_deltas[i] = 6.3*np.power(10.0, -(i+1))

def exponentiate (x):
    return np.power(np.e, x)

def exponentiate_hundredth (x):
    x01 = [0.01*i for i in x]
    return np.power(np.e, x01)

def exp2 (x):
    x2 = [2.0*i for i in x]
    return np.power(np.e, x2)

def identity (x):
    return x

def f_plus_delta (func, x, delta):
    xplusdelta = [i + delta for i in x]
    return func(xplusdelta)

def f_plus_2delta (func, x, delta):
    xplus2delta = [i + 2*delta for i in x]
    return func(xplus2delta)

def f_minus_delta (func, x, delta):
    xminusdelta = [i - delta for i in x]
    return func(xminusdelta)

def f_minus_2delta (func, x, delta):
    xminus2delta = [i - 2*delta for i in x]
    return func(xminus2delta)

def num_derivative (func, x, delta, epsilon):
    fprime = np.zeros(len(x))
    FofX = func(x)
    FplusD = f_plus_delta(func, x, delta)
    Fplus2D = f_plus_2delta(func, x, delta)
    FminusD = f_minus_delta(func, x, delta)
    Fminus2D = f_minus_2delta(func, x, delta)
    
    for i in range(len(x)):
        fprime[i] = (1/12/delta)*(Fminus2D[i] - Fplus2D[i] + 8*FplusD[i] - 8*FminusD[i]) + (epsilon/delta)*FofX[i] + (epsilon/30)*np.power(delta, 4)
    
    return fprime

#Calculates the "true" test values of e^(x) and e^(0.01x)
true_derivative_of_e_to_x = exponentiate(x)
true_derivative_of_e_to_x01 = [0.01*i for i in exponentiate_hundredth(x)]

#calculates the numerical derivatives of e^(x) and e^(0.01x) respectively
numerical_derivative_of_e_to_x = num_derivative (exponentiate, x, optimal_double_delta, double_epsilon)
numerical_derivative_of_e_to_x01 = num_derivative (exponentiate_hundredth, x, optimal_double_delta, double_epsilon)

#calculates the differences between the "true" and numerical derivatives
diff1 = [i - j for i,j in zip(true_derivative_of_e_to_x, numerical_derivative_of_e_to_x)]
diff2 = [i - j for i,j in zip(true_derivative_of_e_to_x01, numerical_derivative_of_e_to_x01)]


#Prints Data Values
print("x values:", x)
print()

print("True derivative of e^(x):" '\n', true_derivative_of_e_to_x)
print("Numerical derivative of e^(x)" '\n', numerical_derivative_of_e_to_x)
print("Value differences:" '\n', diff1)
print("Mean Error in Evaluating e^(x):" '\n', np.mean(diff1))
print()

print("True derivative of e^(0.01x):" '\n', true_derivative_of_e_to_x01)
print("Numerical derivative of e^(0.01x)" '\n', numerical_derivative_of_e_to_x01)
print("Value differences:" '\n', diff2)
print("Mean Error in Evaluating e^(0.01x):" '\n', np.mean(diff2))
print()

#Calculates the mean error in the derivatives (of e^(x) and e^(0.01x) respectively) for varying values of Delta
mean_diff1_varied = np.zeros(len(double_deltas))
mean_diff2_varied = np.zeros(len(double_deltas))
num_deriv_eto_x_varied_delta = np.zeros(len(x))
#num_deriv_eto_x_varied_delta = np.zeros(len(double_deltas))
for i in range(len(double_deltas)):
    num_deriv_eto_x_varied_delta = num_derivative (exponentiate, x, double_deltas[i], double_epsilon)
    mean_diff1_varied[i] = np.mean([true_derivative_of_e_to_x - j for j in num_deriv_eto_x_varied_delta])

num_deriv_eto_01x_varied_delta = np.zeros(len(x))
for i in range(len(double_deltas)):
    num_deriv_eto_01x_varied_delta = num_derivative (exponentiate_hundredth, x, double_deltas[i], double_epsilon)
    mean_diff2_varied[i] = np.mean([true_derivative_of_e_to_x01 - j for j in num_deriv_eto_01x_varied_delta])

print("Mean error in numerical derivative value (of e^x) for deltas ranging from 6.3e-0 to 6.3e-15:" '\n', 
      mean_diff1_varied, '\n')
print("Mean error in numerical derivative value (of e^0.01x) for deltas ranging from 6.3e-0 to 6.3e-15:" '\n',
      mean_diff2_varied, '\n')
print("Conclusion: the roughly optimal values of Delta computed for problem 1 are indeed roughly optimal.")


######################## The following are just my tests I did while coding
"""
#Varies the value of Delta to see which minimizes error in the numerical derivatives
print(double_deltas)
num_deriv_eto_x_varied_delta = np.zeros(len(double_deltas))
for i in range(len(double_deltas)):
    num_deriv_eto_x_varied_delta[i] = num_derivative (exponentiate, x, double_deltas[i], double_epsilon)

num_deriv_eto_01x_varied_delta = np.zeros(len(double_deltas))
for i in range(len(double_deltas)):
    num_deriv_eto_01x_varied_delta[i] = num_derivative (exponentiate, x, double_deltas[i], double_epsilon)

#Calculates the mean error in the derivatives (of e^(x) and e^(0.01x) respectively) for varying values of Delta
mean_diff1_varied = np.zeros(len(double_deltas))
mean_diff2_varied = np.zeros(len(double_deltas))
for j in range(len(double_deltas)):
    mean_diff1_varied[j] = np.mean([true_derivative_of_e_to_x - i for i in num_deriv_eto_x_varied_delta[j]])
    mean_diff2_varied[j] = np.mean([true_derivative_of_e_to_x01 - i for i in num_deriv_eto_01x_varied_delta[j]])

print(mean_diffq_varied)
"""

""" THIS WAS JUST A TEST OF e^(2x)
#Calculates the true derivative of e^(2x)
true_derivative_of_e_to_x2 = [2*i for i in exp2(x)]

#calculates the numerical derivatives of e^(2x)
numerical_derivative_of_e_to_x2 = num_derivative (exp2, x, optimal_double_delta, double_epsilon)
diff3 = [i - j for i,j in zip(true_derivative_of_e_to_x2, numerical_derivative_of_e_to_x2)]

print(true_derivative_of_e_to_x2)
print(numerical_derivative_of_e_to_x2)
print(diff3)
print(np.mean(diff3))
"""

""" THESE JUST TEST MY DELTA FUNCTIONS WHICH WORK FINE
print(f_plus_delta(identity, x, 2))
print(f_plus_2delta(identity, x, 2))
print(f_minus_delta(identity, x, 2))
print(f_minus_2delta(identity, x, 2))
"""

