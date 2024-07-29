# -*- coding: utf-8 -*-
"""
Inlämning 4
@author: danie
namn: Daniel Claesson
YouTube länk: https://youtu.be/etbBWa_B-g0
"""
#%%
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%% Uppgift 1a, plotta som ringar
T = np.array([20.5, 32.7, 51.0, 73.2, 95.7]) # Temperatur
R = np.array([765, 826, 873, 942, 1032])    # Resistans

fig, ax = plt.subplots()
ax.plot(T, R, 'o')
#%% Uppgift 1b, linjär anpassning till mätdatan (1a ordningen)
fit = pol.polyfit(T, R, 1)  # första gradens polynom anpassning
print(fit) # offset=702.17, slope=3.949
m = fit[0]  # offset
k = fit[1]  # lutning
x = np.arange(10,110)   # Return evenly spaced values within a given interval
y = k * x + m

fig, ax = plt.subplots()
ax.plot(T, R, 'o')
ax.plot(x, y)
ax.set_xlabel('Temperatur, T')
ax.set_ylabel('Resistans, R')
#%% Uppgift 1c, uppskatta resistans för T=100
T_100 = k*100+m
print(T_100) #1041.7
#%% Uppgift 2a, mäta pendeltid och anpassa en potensfunktion
# T=a0*L**a1. Jag mäter periodtid T för olika längder L.
# Inspiration från sidan 176.
L = np.array([15.5, 26, 40, 46, 63, 86.5, 107.5, 124]) #pendel längd cm
T = np.array([0.7893, 0.9767, 1.257, 1.341, 1.572, 1.837, 2.054, 2.183]) #pendeltid s

def modellfunktion(L, a):
    """Returnerar T sfa L, baserat på a[0] och a[1]"""
    return a[0] * L**a[1]

def residual(a, T, L):
    """Returnerar residualen map T, mellan mätta värden och modell."""
    return T - modellfunktion(L, a)
    
a0 = [1, 1]     #startgissning
a, q = opt.leastsq(residual, a0, (T, L))    #minimerar residualen mha least squares
print(f'Parametrar: a0={a[0]:.4f}, a1={a[1]:.4f}')

#%% Uppgift 2b, plotta mätdata och modelldata.
#x = np.linspace(0, 150, 150)
x = np.linspace(0, 150, 150)
y = a[0]*x**a[1]
fig, ax = plt.subplots()
ax.scatter(L, T)
ax.plot(x, y, '#FFA500')
ax.set_xlabel('Replängd [cm]')
ax.set_ylabel('Pendlingstid [s]')
ax.set_title('Uppgift 2, Pendel')

#%% Uppgift 2c: Vilken L ger T = 1 s?
# sidan 166-167 för inspiration
f = lambda x : a[0]*x**a[1] # f(x)=a0*L**a1
g = lambda x : f(x) - 1     # g(x) = f(x) - 1
r = opt.root(g,1)   # startgissning = 1
print(f'Längden L (cm) som ger tiden T 1s: {r.x}') # 25.7 cm
#%% Uppgift 3a, plottar datan
S = np.array([1.5, 2, 3, 4, 8]) #Koncentration, mol
v = np.array([0.21, 0.25, 0.28, 0.33, 0.44]) # Reaktionshastighet

fig, ax = plt. subplots()
ax.plot(S, v, 'o')
ax.set_xlabel('Koncentration, [mol]')
ax.set_ylabel('Reaktionshastighet, [mg/min]')
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
#%% Uppgift 3b, anpassa rationell funktion till datan och plotta
def modellfunktion(S, a):
    return (a[0] * S) / (1 + a[1] * S)

def residual(a, S, v):
    return v - modellfunktion(S, a)

a0 = [1, 1]     #startgissning
a, q = opt.leastsq(residual, a0, (S, v))
print(f'Parametrar: a0={a[0]:.4f}, a1={a[1]:.4f}')

x = np.linspace(0, 10, 1000)
y = (a[0]*x)/(1+a[1]*x)

fig, ax = plt.subplots()
ax.plot(S, v, 'o')
ax.set_xlabel('Koncentration, [mol]')
ax.set_ylabel('Reaktionshastighet, [mg/min]')
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
ax.plot(x, y, '#FFA500')
# %% Uppgift 4a, plotta innehållet i population.txt
path = r'C:/Users/danie/Documents/GitHub/Matematisk-Modellering-MAU/Uppgift4/population.txt'
data = np.loadtxt(path)
x = np.linspace(0,len(data),len(data))

fig, ax = plt.subplots()
ax.plot(x, data)
ax.set_xlabel('År, baseline 1951')
ax.set_ylabel('Population, miljarder')
# %%Uppgift 4b, anpassa en exponentialfunktion till datan
def modellfunktion(a, x):
    return a[0] * np.exp(a[1] * x) # returnerar population sfa tid (x)

def residual(a, x, data):
    return data - modellfunktion(a, x)

a0 = [3, 0.01] #initialgissning
a, q = opt.leastsq(residual, a0, (x, data))
print(f'Parametrar: a0={a[0]:.4e}, a1={a[1]:.4e}')

# skapar en ny x-axel:
x2 = np.linspace(0, 150, 150)
y2 = a[0] * np.exp(a[1] * x2)

fig, ax = plt.subplots()
ax.plot(x, data, 'b--')
ax.set_xlabel('År, baseline 1951')
ax.set_ylabel('Population, miljarder')
ax.plot(x2, y2, '#FFA500')
#%% Uppgift 4c, när är befolkningen > 15 miljarder? Zooma i grafen