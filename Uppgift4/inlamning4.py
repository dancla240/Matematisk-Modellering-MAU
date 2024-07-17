# -*- coding: utf-8 -*-
"""
Inlämning 4
@author: danie
namn: Daniel Claesson
YouTube länk: 
"""
#%%
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%% Uppgift 1a
T = np.array([20.5, 32.7, 51.0, 73.2, 95.7])
R = np.array([765, 826, 873, 942, 1032])

fig, ax = plt.subplots()
ax.plot(T, R, 'o')
#%% Uppgift 1b, linjär anpassning till mätdatan (1a ordningen)
fit = pol.polyfit(T, R, 1)
print(fit) # offset=702.17, slope=3.949
m = fit[0]
k = fit[1]
x = np.arange(10,110)
y = k * x + m

fig, ax = plt.subplots()
ax.plot(T, R, 'o')
ax.plot(x, y)
ax.set_xlabel('Temperatur, T')
ax.set_ylabel('Resistans, R')
#%% Uppgift 1c, uppskatta resistans för T=100
T_100 = k*100+m
print(T_100) #1041.7
#%% Uppgift 2a, mäta pendeltid
L = np.array([15.5, 26, 40, 46, 63, 86.5, 107.5, 124]) #pendel längd cm
T = np.array([0.7893, 0.9767, 1.257, 1.341, 1.572, 1.837, 2.054, 2.183]) #pendeltid s

def modellfunktion(L, a):
    """Returnerar T sfa L, baserat på a"""
    return a[0] * L**a[1]

def res(a, T, L):
    return T - modellfunktion(L, a)
    
a0 = [0.3, 0.6]     #startgissning
a, q = opt.leastsq(res, a0, (T, L))
print('Parametrar', a)

# %%
