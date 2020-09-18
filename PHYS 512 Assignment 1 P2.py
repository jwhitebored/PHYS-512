# -*- coding: utf-8 -*-
"""
PHYS 512
Assignment 1 Problem 2
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


#Extracted Lakeshore Data
"""
lakeshore = pd.read_csv("lakeshore_mod.txt", sep = '\t', header = None)
print(lakeshore)

lakeshore_array = lakeshore.values.tolist()

Temp = lakeshore[0].tolist() #Kelvin
Volt = lakeshore[1].tolist() #Volts
dvdt = lakeshore[2].tolist() #mV/K
"""

#I always hard-code my data in my program (had issues in the past with myself/partners losing data files)
Temp = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 65.0, 70.0, 75.0, 77.35, 80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 273.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0]
Volt = [1.6442900000000003, 1.6429900000000002, 1.64157, 1.6400299999999999, 1.63837, 1.6366, 1.6347200000000002, 1.6327399999999999, 1.63067, 1.62852, 1.62629, 1.624, 1.62166, 1.6192799999999998, 1.61687, 1.61445, 1.612, 1.60951, 1.60697, 1.60438, 1.6017299999999999, 1.59902, 1.59626, 1.59344, 1.59057, 1.58764, 1.58465, 1.5784799999999999, 1.57202, 1.5653299999999999, 1.55845, 1.55145, 1.54436, 1.53721, 1.53, 1.52273, 1.5154100000000001, 1.49698, 1.47868, 1.46086, 1.44374, 1.42747, 1.4120700000000002, 1.39751, 1.38373, 1.37065, 1.3582, 1.3463200000000002, 1.33499, 1.32416, 1.3138100000000001, 1.3039, 1.29439, 1.28526, 1.27645, 1.26794, 1.25967, 1.2516100000000001, 1.2437200000000002, 1.23596, 1.2283, 1.2207, 1.2131100000000001, 1.2054799999999999, 1.197748, 1.181548, 1.1627969999999999, 1.140817, 1.125923, 1.119448, 1.115658, 1.11281, 1.1104209999999999, 1.1082610000000002, 1.106244, 1.104324, 1.102476, 1.100681, 1.09893, 1.097216, 1.095534, 1.0938780000000001, 1.092244, 1.090627, 1.089024, 1.085842, 1.082669, 1.0794920000000001, 1.076303, 1.073099, 1.0698809999999999, 1.06665, 1.063403, 1.060141, 1.0568620000000002, 1.048584, 1.040183, 1.031651, 1.0275940000000001, 1.0229840000000001, 1.014181, 1.005244, 0.986974, 0.968209, 0.949, 0.9293899999999999, 0.909416, 0.889114, 0.868518, 0.847659, 0.8265600000000001, 0.805242, 0.78372, 0.762007, 0.740115, 0.718054, 0.6958340000000001, 0.673462, 0.650949, 0.628302, 0.6211409999999999, 0.605528, 0.582637, 0.559639, 0.536542, 0.513361, 0.49010600000000004, 0.46676000000000006, 0.443371, 0.41996000000000006, 0.396503, 0.373002, 0.349453, 0.325839, 0.302161, 0.278416, 0.254592, 0.23069699999999999, 0.206758, 0.182832, 0.15900999999999998, 0.13548, 0.112553, 0.090681]
dvdt = [-12.5, -13.6, -14.8, -16.0, -17.1, -18.3, -19.3, -20.3, -21.1, -21.9, -22.6, -23.2, -23.6, -24.0, -24.2, -24.4, -24.7, -25.1, -25.6, -26.2, -26.8, -27.4, -27.9, -28.4, -29.0, -29.6, -30.2, -31.6, -32.9, -34.0, -34.7, -35.2, -35.6, -35.9, -36.2, -36.5, -36.7, -36.9, -36.2, -35.0, -33.4, -31.7, -29.9, -28.3, -26.8, -25.5, -24.3, -23.2, -22.1, -21.2, -20.3, -19.4, -18.6, -17.9, -17.3, -16.8, -16.3, -15.9, -15.6, -15.4, -15.3, -15.2, -15.2, -15.3, -15.6, -17.0, -21.1, -20.8, -9.42, -4.6, -3.19, -2.58, -2.25, -2.08, -1.96, -1.88, -1.82, -1.77, -1.73, -1.7, -1.69, -1.64, -1.62, -1.61, -1.6, -1.59, -1.59, -1.59, -1.6, -1.61, -1.61, -1.62, -1.63, -1.64, -1.64, -1.67, -1.69, -1.72, -1.73, -1.75, -1.77, -1.8, -1.85, -1.9, -1.94, -1.98, -2.01, -2.05, -2.07, -2.1, -2.12, -2.14, -2.16, -2.18, -2.2, -2.21, -2.23, -2.24, -2.26, -2.27, -2.28, -2.28, -2.29, -2.3, -2.31, -2.32, -2.33, -2.34, -2.34, -2.34, -2.35, -2.35, -2.36, -2.36, -2.37, -2.38, -2.39, -2.39, -2.39, -2.39, -2.37, -2.33, -2.25, -2.12]

#Sorts both Volt and Temp lists together according to the ascending order of the first list (Volt)
VoltTempSortedZip = sorted(zip(Volt,Temp))                          #Returns sorted zip
VoltTempTuple = zip(*VoltTempSortedZip)                             #Returns a tuple = (VoltSorted, TempSorted)
Volt, Temp = [ list(tuple) for tuple in VoltTempTuple]                    #The sorted Volt list


#Produces lists of every other value for Voltage and Temp that will be used to estimate spline error
coarseVolt = np.zeros(64)
coarseTemp = np.zeros(64)
for i in range(64):
    coarseVolt[i] = Volt[2*i]
    coarseTemp[i] = Temp[2*i]
  
#Making the splines:
splineDomain = np.linspace(0.08, 1.7, 1000)
spline = interpolate.splrep(Volt, Temp)                             #Solves for the spline Values
splineCoarse = interpolate.splrep(coarseVolt, coarseTemp)           #Solves for the spline Values

splineTemp = interpolate.splev(splineDomain, spline)                #Spline for graphical purposes
splineTempCoarse = interpolate.splev(splineDomain, splineCoarse)    #Spline for graphical purposes

TempFit = interpolate.splev(Volt, spline)                           #Spline for Residuals
coarseTempFit = interpolate.splev(coarseVolt, spline)               #Spline for Residuals

####################################### PLOT ######################################################
fig, ax = plt.subplots(2, figsize=(10,10), gridspec_kw={'height_ratios': [2, 1]})

#ax = f1.add_subplot(2,1,1)
ax[0].plot(Volt, Temp, "m*", label = 'Data')
ax[0].plot(splineDomain, splineTemp, 'b', label = 'Cubic Spline Fit')
ax[0].plot(splineDomain, splineTempCoarse, 'r', label = 'Cubic Spline of Every Other Datum')
ax[0].set_xlabel("Voltage (V)")
ax[0].set_ylabel("Temperature (K)")
ax[0].set_title("Temperature of a Lakeshore 670 Diode as a Function of Voltage")

##Residuals
diff1 = [i-j for i,j in zip(TempFit, Temp)]
diff2 = [i-j for i,j in zip(coarseTempFit, coarseTemp)]

#ax2.yaxis.tick_right()
ax[1].plot(Volt, diff1, 'bo')
ax[1].plot(coarseVolt, diff2, 'ro')
ax[1].plot(Volt, [0]*len(Volt), color = 'black')
ax[1].set_xlabel("Voltage")
ax[1].set_ylabel("Spline Residuals")
plt.show()

#################################### Interpolator #################################################
def interpolatedTemp (x):
    interTemp = interpolate.splev(x, spline)
    error = np.abs(interpolate.splev(x, splineCoarse) - interpolate.splev(x, spline))
    return print("Temp:", interTemp, "\nApproximate Error:", error)

interpolatedTemp(0.8) #Test seems about right!







