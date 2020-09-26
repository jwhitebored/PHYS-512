# -*- coding: utf-8 -*-
"""
PHYS 512
Assignment 2 Problem 2
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
from scipy.special import legendre as LEG
############################# Functions #######################################

#finds the coeffictients to the Chebyshev polynomial of order 'ord' that model function 'fun' from 'xleft' to 'xright'
def cheb_fit(fun,xleft,xright,ord):
    x=np.linspace(xleft,xright,ord+1)
    y=fun(x)
    mat=np.zeros([ord+1,ord+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(1,ord):
        mat[:,i+1]=2*x*mat[:,i]-mat[:,i-1]
    coeffs=np.dot(np.linalg.inv(mat),y)
    return coeffs

#given Chebyshev polynomial coefficients, this function takes the first 'order' of them, and fits 
def trunc_cheb_fit(coeffs, order, arrayx):
    coeffs=np.delete(coeffs,np.s_[order+1:])                                    #truncates the coeffs list
    assert(len(coeffs)==order+1)
    
    polyMatrix = np.zeros([len(arrayx), order+1])                               #makes matrices with 'arrayx'
    polyYvalues = np.zeros(len(arrayx))                                         #rows and 'order' columnes
    polyMatrix[:,0] = 1
    polyMatrix[:,1] = arrayx
    
    for i in range(len(arrayx)):
        for j in range(1, order):
            polyMatrix[i,j+1] = 2*arrayx[i]*polyMatrix[i,j] - polyMatrix[i,j-1]
            
        polyYvalues[i] = np.dot(polyMatrix[i,:], coeffs)
    return polyYvalues

###############################################################################
    
############################### Chebyshev Fitting #############################
#Chebyshev fitting works on -1 to 1 so I had to shift my log2 function to fit it properly
def shiftLog2(x):
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=x[i]+1.5
    return np.log2(y)    

fun = shiftLog2
computationOrder = 50
leftbound = -1
rightbound = 1

truncationOrder = 7
chebCoeffs = cheb_fit(fun,leftbound,rightbound,computationOrder)                #Note the order of the polynomial is 'len(coeffs)+1'
                                                                                #due to x^0=1
xCheb=np.linspace(leftbound,rightbound-1.5,101)
chebfitYvalues = trunc_cheb_fit(chebCoeffs,truncationOrder,xCheb)

############################## Legendre Fitting ###############################
legcoeffs = np.polynomial.legendre.legfit(xCheb,fun(xCheb),computationOrder)

def legendre_poly(coeffs, arrayx, order):
    leg = np.zeros([len(arrayx), order+1])
    legYvalues = np.zeros(len(arrayx))
    for i in range(len(arrayx)):
        for j in range(order+1):
            leg[i,j] = LEG(j)(arrayx[i])
        legYvalues[i] = np.dot(leg[i,:], coeffs)
    return legYvalues

legfitYvalues = legendre_poly(legcoeffs, xCheb, computationOrder)
############################# PLOTS ###########################################
xplot = [i+1.5 for i in xCheb]
fig, ax = plt.subplots(2, figsize=(10,10), gridspec_kw={'height_ratios': [2, 1]})

#ax = f1.add_subplot(2,1,1)
ax[0].plot(xplot, fun(xCheb), color = 'black', label = 'np.log2(x)')
ax[0].plot(xplot, chebfitYvalues, "m*", label = 'Chebyshev Fit')
ax[0].plot(xplot, legfitYvalues, 'b.', label = 'Legendre Fit')
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Chebyshev and Legendre Fits of np.log2(x)")
ax[0].legend(loc="lower right")

##Residuals
diffcheb = [i-j for i,j in zip(chebfitYvalues, np.log2(xplot))]
diffleg = [i-j for i,j in zip(legfitYvalues, np.log2(xplot))]

#ax2.yaxis.tick_right()
ax[1].plot(xplot, [0]*len(xplot), color = 'black')
ax[1].plot(xplot, diffcheb, 'm*')
ax[1].plot(xplot, diffleg, 'b.')
ax[1].set_xlabel("x")
ax[1].set_ylabel("Chebyshev and Legendre Fit Residuals")
#plt.savefig('cheb_legend_fits.png')
plt.show()

#################### Question Responses #######################################
"""
1. How many terms do you need to get error < 10^-6?

For the life of me I can't get my error that low. After trying over a hundred
orders for the Chebyshev polynomials, I've found the error gets ludicrously
worse for order > 50. I've read a fair amount on Chebyshev polys at this point
and everything says this shouldn't happen, and that the error should remain
practically fixed for high enough order (regardless of where I truncate it past
said order). Before heavily modifying/improving it, I copied verbatum the code 
from class, and noticed that my laptop was producing different values than what 
were achieved in class. This was specifically for fitting np.sin with a 51st 
degree Chebyshev in lecture 4. Someone on slack seemed to have the same
issue that they couldn't resolve and it was suggested machine round-off error
was getting the best of them. Perhaps this is the case

2. Compare the max and rms error for both fits.

For the reasons above, the legendre fit has lesser values for both max and rms
error. This is blindingly evident in my plot for this problem (which I'm very
proud turned out so nice).
"""




















