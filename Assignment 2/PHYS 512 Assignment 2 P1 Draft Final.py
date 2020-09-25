# -*- coding: utf-8 -*-
"""
PHYS 512
Assignment 2 Problem 1
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
################################################################################

##### Variable-Step-Size-Integrator from Class (With Repetative Function Calls)#
def lorentz(x):
    return 1/(1+x**2)

def lazy_integrate_step(fun,x1,x2,tol):
    #print('integrating from ',x1,' to ',x2)
    x=np.linspace(x1,x2,5)
    y=fun(x)
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    neval=len(x) #let's keep track of function evaluations
    
    if myerr<tol:
        return area2, myerr, neval
    
    else:
        xm=0.5*(x1+x2)
        a1, leftErr, leftEval= lazy_integrate_step(fun,x1,xm,tol/2)
        a2, rightErr, rightEval = lazy_integrate_step(fun,xm,x2,tol/2)
        sumError = leftErr + rightErr
        totEval = neval + leftEval + rightEval

        return a1+a2, sumError, totEval

################################################################################

##### Variable-Step-Size-Integrator (Without Repetative Function Calls)#########
def integrate_step(fun,x1,x2,tol, XLIST = np.array([]), YLIST = np.array([])):
#    print('integrating from ',x1,' to ',x2)
    x=np.linspace(x1,x2,5)
    y=np.zeros(len(x))
    
    for i in range(len(x)):
        if x[i] in XLIST:                           #if the point x[i] for this iteration is already present
            index = np.where(XLIST == x[i])[0]      #in the ongoing list of domain points for the entire 
            y[i] = YLIST[index]                     #integral (XLIST), then y(x[i]) has already been calculated 
                                                    #and is not calculated again. Instead it is given the 
                                                    #precalculated value from the ongoing list of y-values for 
                                                    #the entire integral: YLIST
        
        else:                                       #if the point x[i] is not in XLIST, then y(x[i]) is not in
            y[i] = fun(x[i])                        #YLIST, so y(x[i]) is calculated here, and then XLIST and
            XLIST = list(np.append(XLIST, x[i]))    #YLIST are updated to inclued x[i] and y(x[i]) in the
            YLIST = list(np.append(YLIST, y[i]))    #correct order of x[i] ascending
            XLIST, YLIST = [list(tuple) for tuple in zip(*sorted(zip(XLIST, YLIST)))]
            XLIST = np.array(XLIST)
            YLIST = np.array(YLIST)
    
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    neval=len(YLIST)                                #By my above book-keeping, the number of times y(x) is 
                                                    #evaluated is simply len(YLIST)
#    print("y(x) evaluations so far:", len(YLIST))
    if myerr<tol:                                   #If error is tolerable, returns area for this portion
        return area2, myerr, neval, XLIST, YLIST    #of the integral
    
    else:                                           #If error is not tolerable, computes integral of each half
        xm=0.5*(x1+x2)                              #of the domain separately, doubling the precision. This is 
                                                    #done via recurssion
        a1, leftErr, leftEval, XLIST, YLIST= integrate_step(fun,x1,xm,tol/2, XLIST, YLIST)
        a2, rightErr, rightEval, XLIST, YLIST = integrate_step(fun,xm,x2,tol/2, XLIST, YLIST)
        sumError = leftErr + rightErr
        totEval = len(YLIST)
        return a1+a2, sumError, totEval, XLIST, YLIST
    
###############################################################################

#y=e^x integration with improved integrator
EXPstart = -1
EXPstop = 1
EXPf,EXPerr,EXPneval,EXPxlist,EXPylist=integrate_step(np.exp,EXPstart,EXPstop,1e-3)
EXPtrue=np.exp(EXPstop)-np.exp(EXPstart)

#y=e^x integration with integrator from class
lazyEXPf,lazyEXPerr,lazyEXPneval=lazy_integrate_step(np.exp,EXPstart,EXPstop,1e-3)

print("Numerical Integral of e^x from -1 to 1:", EXPf)
print("Function Evaluations for Integral of e^x (Improved way):", EXPneval)
print("Function Evaluations for Integral of e^x ('Lazy' way):", lazyEXPneval, '\n')


#Lorentzian integration with improved integrator
LORENTZstart = -1
LORENTZstop = 1
LORENTZf,LORENTZerr,LORENTZneval,LORENTZxlist,LORENTZylist=integrate_step(lorentz,LORENTZstart,LORENTZstop,1e-3)

#Lorentzian integration with integrator from class
lazyLORENTZf,lazyLORENTZerr,lazyLORENTZneval=lazy_integrate_step(lorentz,LORENTZstart,LORENTZstop,1e-3)

print("Numerical Integral of the Lorentzian from -1 to 1:", LORENTZf)
print("Function Evaluations for Integral of the Lorentzian (Improved way):", LORENTZneval)
print("Function Evaluations for Integral of the Lorentzian ('Lazy' way):", lazyLORENTZneval, '\n')


#sin(x) integration with improved integrator
SINstart = 0
SINstop = np.pi
SINf,SINerr,SINneval,SINxlist,SINylist=integrate_step(np.sin,SINstart,SINstop,1e-3)

#sin(x) integration with integrator from class
lazySINf,lazySINerr,lazySINneval=lazy_integrate_step(np.sin,SINstart,SINstop,1e-3)

print("Numerical Integral of sin(x) from 0 to pi:", SINf)
print("Function Evaluations for Integral of sin(x) (Improved way):", SINneval)
print("Function Evaluations for Integral of sin(x) ('Lazy' way):", lazySINneval, '\n')

############### Plot of Sampled Points for e^x ################################

fig, ax = plt.subplots(1, figsize=(10,10))
#ax = f1.add_subplot(2,1,1)
#ax.plot(np.linspace(EXPstart, EXPstop, 100), np.exp(np.linspace(EXPstart, EXPstop, 100)), 'r.')
ax.plot(EXPxlist, EXPylist, "m*", markersize = 15, label = 'sample points')
ax.set_xlabel("x")
ax.tick_params(axis='x', labelsize=15)
ax.set_ylabel("y=e^x")
ax.tick_params(axis='y', labelsize=15)
ax.set_title("Points Sampled to Compute Integral of e^x")
plt.show()


############################# SCRAP ###########################################
"""
def inlist(array, item):
    if item in array:
        itemindex = np.where(array==item)
        mylist = itemindex[0].tolist()
        return mylist                                                  #Returns index(es) of 'item' in 'array'
    else:
        return None#print(item, " not in list")
"""




















