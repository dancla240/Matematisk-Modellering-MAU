# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:05:46 2024

@author: danie
"""
#%% Importerar librarys
import numpy as np
import matplotlib.pyplot as plt
#%% Uppgift 1, läser in data
mens_length = np.loadtxt('men_length.txt')
women_length = np.loadtxt('women_length.txt')
#%% Uppgift 1, printar men
print('Men max:', np.max(mens_length))
print('Men min:', np.min(mens_length))
print('Men mean:', round(np.mean(mens_length), 1))
print('Men stddev:', round(np.std(mens_length), 1))
# %% Uppgift 1 plottar men
fig, ax = plt.subplots()
ax.hist(mens_length, bins=20)
ax.set_ylabel('lenght')
ax.set_xlabel('frequence')
ax.set_title('Histogram Mens Lenght')

#%% Uppgift 1, printar women
print('Women max:', np.max(women_length))
print('Women min:', np.min(women_length))
print('Women mean:', round(np.mean(women_length), 1))
print('Women stddev:', round(np.std(women_length), 1))
# %% Uppgift 1 plottar women
fig, ax = plt.subplots()
ax.hist(women_length, bins=20)
ax.set_ylabel('lenght')
ax.set_xlabel('frequence')
ax.set_title('Histogram Womens Lenght')
# %% Uppgift 2
scb_age = np.loadtxt('scb.txt')
klass_granser = np.arange(-0.5, 104.6, 5)
klass_mitt = np.arange(2, 104, 5)

medel = np.sum(klass_mitt * scb_age)/np.sum(scb_age)
print('Medel:', round(medel,1))

fig, ax = plt.subplots()
ax.stairs(values=scb_age, edges=klass_granser, fill=True)
ax.set_xlabel('Ålder')
ax.set_ylabel('Frekvens')
#%% beräkna antalet personer > 70 år
mask = klass_mitt > 70
selected_values = scb_age[mask]
antal = np.sum(selected_values)
print('Antal män >= 70 år:', antal)
#%% Uppgift 3a, plottar signalen och en medelvärdesbildad signal
y = np.loadtxt('signal.txt')
ym = np.zeros(len(y)) # fördimensionering
fig, ax = plt.subplots()
ax.plot(y)
k = 5
n = len(y)
for i in range(n): # loop över element
    n1 = np.max([0,i-k]) # n1 aldrig mindre än 0
    n2 = np.min([n,i+k+1]) # n2 aldrig större än n
    ym[i] = np.mean(y[n1:n2]) # medelvärde i fönster
ax.plot(ym,'r')
ax.set_xlim([0,n])
ax.set_ylabel(r'T i $^o$C',fontsize=14)
ax.tick_params(labelsize=14)
#%% Uppgift 3b, median istället för medel
y = np.loadtxt('signal.txt')
ym = np.zeros(len(y)) # fördimensionering
fig, ax = plt.subplots()
ax.plot(y)
k = 5
n = len(y)
for i in range(n): # loop över element
    n1 = np.max([0,i-k]) # n1 aldrig mindre än 0
    n2 = np.min([n,i+k+1]) # n2 aldrig större än n
    ym[i] = np.median(y[n1:n2]) # medelvärde i fönster
ax.plot(ym,'r')
ax.set_xlim([0,n])
ax.set_ylabel(r'T i $^o$C',fontsize=14)
ax.tick_params(labelsize=14)
#%% Uppgift 4
mass = 5000 + 100 * np.random.randn(100000)
v = 400 + 70 * np.random.randn(100000)
W = 1/2 * mass * v**2
fig, ax = plt.subplots()
ax.hist(W, bins=20)
#%% Uppgift 5
# Räta linjens ekvation: y = k*x + m
# k = dy/dx, lutningen på linjen
# m = offset, vart på y-axeln linjen skär
k = (4-(-1))/(2-(-2))   # k = 5/4 = 1.25
m = -1 -1.25*(-2)       # m = y - k*x = 1.5
# dvs, räta linjens ekvation: y = 1.25*x + 1.5

