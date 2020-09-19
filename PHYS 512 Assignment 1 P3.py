# -*- coding: utf-8 -*-
"""
PHYS 512
Assignment 1 Problem 3
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
from scipy import interpolate

###################################################################################################
###################################### Cos(x) Fitting #############################################
###################################################################################################

#simulated data values:
xdata = np.linspace((-0.5*np.pi), (0.5*np.pi), 16)
ydata = np.cos(xdata)

########################################### Cubic Spline Stuff #############################
#Making the Cubic Spline
spline = interpolate.splrep(xdata, ydata)

#Values for plotting the cubic spline
splineDomain = np.linspace(xdata[0], xdata[-1], 1000)
splineYData = interpolate.splev(splineDomain, spline)

#Values for calculating the differences between the spline fit and the true values
ydataSplineFit = interpolate.splev(xdata, spline)

################################## Rational Function Fit Stuff #############################
#defining the rational function from class:
def rat_eval(p,q,x):
    top = 0
    for i in range(len(p)):
        top = top + p[i]*x**i
    bot = 1
    for i in range(len(q)):
        bot = bot + q[i]*x**(i+1)
    return top/bot

#rational function fitting method from class:
def rat_fit(x,y,n,m):
    assert(len(x) == n+m-1)
    assert(len(y) == len(x))
    mat = np.zeros([n+m-1, n+m-1])                  #Makes a matrix size [rows, columns] of zeros
    for i in range(n):
        mat[:,i] = x**i                             #replaces [all rows, column i] with with the column vector [x**i] up to nth column
    for i in range(1,m):
        mat[:,i-1+n] = -y*x**i                      #replaces [all rows, column i-1 +n] with the column vector [-yx**i] after nth column
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q
    
#Calculating the Rational Function Fit
n=8
m=9
p,q = rat_fit(xdata,ydata,n,m)

#Values for plotting the cubic spline
ratDomain = np.linspace(xdata[0], xdata[-1], 1000)
ratYData = rat_eval(p,q,ratDomain)

#Values for calculating the differences between the rational fit and the true values
ydataRatFit = rat_eval(p,q,xdata)

############################################### Plots for Cos(x) ######################################

fig, ax = plt.subplots(2, figsize=(10,10), gridspec_kw={'height_ratios': [2, 1]})

#ax = f1.add_subplot(2,1,1)
ax[0].plot(xdata, ydata, "m*", label = 'Data')
ax[0].plot(splineDomain, splineYData, 'b', label = 'Cubic Spline Fit')
ax[0].plot(ratDomain, ratYData, 'r', label = 'Rational Fit')
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Cubic Spline and Rational Fits of y=cos(x)")

##Residuals
diff1 = [i-j for i,j in zip(ydataSplineFit, ydata)]
diff2 = [i-j for i,j in zip(ydataRatFit, ydata)]

#ax2.yaxis.tick_right()
ax[1].plot(xdata, diff1, 'bo')
ax[1].plot(xdata, diff2, 'ro')
ax[1].plot(xdata, [0]*len(xdata), color = 'black')
ax[1].set_xlabel("x")
ax[1].set_ylabel("Spline and Rational Fit Residuals")
plt.show()

#I'm happy with both my spline and rational fits on this interval, but one thing I've noticed is that
#by increasing the number of data points used to make the rational fit, the worse the fit becomes near x=0
#It seems to develope a spike there. It reminds me of how the edges of a square wave have tips you can't
#get rid of unless you actually have an infinite amount of terms to express the function, though I doubt
#that's the case here

"""
#Testing how matrix filling works in a for loop
mat1 = np.zeros([10,10])
mat2 = np.zeros([10,10])
x = [1,2,3,4,5,6,7,8,9,10]
for i in range(10):
    mat1[:,i] = i
    mat2[:,i] = x

print(mat1)
print(mat2)
"""
#######################################################################################################
############################################ Lorentzian Fitting #######################################
#######################################################################################################

#simulated data values:
xLdata = np.linspace(-1, 1, 34)
yLdata = np.array([1/(1+i**2) for i in xLdata])
Lorentz = [1/(1+i**2) for i in np.linspace(-1,1,100)]

########################################### Cubic Spline Stuff #############################
#Making the Cubic Spline
splineL = interpolate.splrep(xLdata, yLdata)

#Values for plotting the cubic spline
splineDomainL = np.linspace(xLdata[0], xLdata[-1], 1000)
splineYDataL = interpolate.splev(splineDomainL, splineL)

#Values for calculating the differences between the spline fit and the true values
ydataSplineFitL = interpolate.splev(xLdata, splineL)

################################## Rational Function Fit Stuff #############################
#Calculating the Rational Function Fit
N=34
M=1
P,Q = rat_fit(xLdata,yLdata,N,M)

#Values for plotting the cubic spline
ratDomainL = np.linspace(xLdata[0], xLdata[-1], 1000)
ratYDataL = rat_eval(P,Q,ratDomainL)

#Values for calculating the differences between the rational fit and the true values
ydataRatFitL = rat_eval(P,Q,xLdata)

############################## Plots for Lorentzian 1/(1+x^2) ##############################

fig2, ax2 = plt.subplots(2, figsize=(10,10), gridspec_kw={'height_ratios': [2, 1]})

#ax = f1.add_subplot(2,1,1)
ax2[0].plot(xLdata, yLdata, "m*", label = 'Data')
ax2[0].plot(splineDomainL, splineYDataL, 'b', label = 'Cubic Spline Fit')
ax2[0].plot(ratDomainL, ratYDataL, 'r', label = 'Rational Fit')
ax2[0].plot(np.linspace(-1,1,100), Lorentz, 'g')
ax2[0].set_xlabel("x")
ax2[0].set_ylabel("y")
ax2[0].set_title("Cubic Spline and Rational Fits of y=1/(1+x^2)")

##Residuals
diff1L = [i-j for i,j in zip(ydataSplineFitL, yLdata)]
diff2L = [i-j for i,j in zip(ydataRatFitL, yLdata)]

#ax2.yaxis.tick_right()
ax2[1].plot(xLdata, diff1L, 'bo')
ax2[1].plot(xLdata, diff2L, 'ro')
ax2[1].plot(xLdata, [0]*len(xLdata), color = 'black')
ax2[1].set_xlabel("x")
ax2[1].set_ylabel("Spline and Rational Fit Residuals")
plt.show()

#In this plot, the green curve (the true lortentzian graph) and the red curve(the rational approximation)
#overlap entirely. This is nice, but the only values of N,M where M=N+1 that produce such a good result are 2,3
#and this requires there to be LESS data points to interpolate from (N,M = 2,3 implies 4 data points)! 
#That's a horrible result. I did actually get the rational approximation to work very nicely for the following
#N,M values: [24,2], [26,2], [28,2] which use 25, 27, and 29 data points respectively, and [N<36,M=1]. But all
#of these N,M pairs, got WORSE as the number of data points increased.
#Looking at the P list I believe these results are due to the fact that the expansion of 1/(1+x^2) for |x|<1 is

#       1 - x^2 + x^4 - x^8 + x^16 - x^32 + x^64 - x^128 + x^256 -...

#All to say I'm surprised how bad this result is, and I checked over my code for hours
#(not that it gaurentees anything).

print(P)
print(Q)

#for N,M=35,1 P=[ 9.97000102e-01 -9.16953053e-03 -1.00382580e+00  7.48501600e-02
#  9.92984504e-01 -4.34249196e-01 -9.15164433e-01  1.08110869e+00
#  7.77383152e-01 -1.36070037e+00 -2.50276502e-01  1.51638412e+00
# -9.18426871e-01 -2.83326721e+00  1.71587729e+00  4.20068359e+00
# -1.39774561e+00 -3.61621094e+00  7.19886780e-01  2.13671875e+00
# -1.78497314e-01 -1.15625000e+00 -3.25843811e-01  4.29687500e-01
#  5.63583374e-01 -1.56250000e-02 -4.31777954e-01 -1.56250000e-02
#  2.07420349e-01  0.00000000e+00 -6.48155212e-02  7.81250000e-03
#  9.21344757e-03 -4.88281250e-04]












