# -*- coding: utf-8 -*-
"""
Inlämning 3
Created on Tue Jul  2 20:05:46 2024
@author: danie
namn: Daniel Claesson
YouTube länk: https://youtu.be/nlivYOI4byo
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
ax.set_ylabel('frequence')
ax.set_xlabel('Length [cm]')
ax.set_title('Histogram Mens Lenght')

#%% Uppgift 1, printar women
print('Women max:', np.max(women_length))
print('Women min:', np.min(women_length))
print('Women mean:', round(np.mean(women_length), 1))
print('Women stddev:', round(np.std(women_length), 1))
# %% Uppgift 1 plottar histogram women
fig, ax = plt.subplots()
ax.hist(women_length, bins=20)
ax.set_ylabel('frequence')
ax.set_xlabel('Length [cm]')
ax.set_title('Histogram Womens Lenght')
# %% Uppgift 2, ålder män Sverige
scb_ålder = np.loadtxt('scb.txt')
klass_gränser = np.arange(-0.5, 104.6, 5)   #[0.5, 4.5, 9.5 osv]
klass_mitt = np.arange(2, 104, 5)           #[2, 7, 12 osv]

# beräknar medelåldern för män
medelålder = np.sum(klass_mitt * scb_ålder)/np.sum(scb_ålder)
print('Medelålder:', round(medelålder,1))

fig, ax = plt.subplots()
ax.stairs(values=scb_ålder, edges=klass_gränser, fill=True)
ax.set_xlabel('Ålder')
ax.set_ylabel('Frekvens')
#%% Uppgift 2 forts, beräkna antalet personer > 70 år
mask = klass_mitt > 70
selected_values = scb_ålder[mask]
antal = np.sum(selected_values)
print('Antal män >= 70 år:', antal)
#%% Uppgift 3a, plottar signalen och en medelvärdesbildad signal
y = np.loadtxt('signal.txt')
y_medel = np.zeros(len(y)) # fördimensionering, vektor med nollor i y lång
fig, ax = plt.subplots()
ax.plot(y)
k = 5
n = len(y)
for i in range(n): # loop över element
    n1 = np.max([0,i-k]) # n1 aldrig mindre än 0
    n2 = np.min([n,i+k+1]) # n2 aldrig större än n
    y_medel[i] = np.mean(y[n1:n2]) # medelvärde i fönster
ax.plot(y_medel,'r')
ax.set_xlim([0,n])
ax.set_ylabel(r'T i $^o$C',fontsize=14)
ax.tick_params(labelsize=14)
#%% Uppgift 3b, median istället för medel
y = np.loadtxt('signal.txt')
y_median = np.zeros(len(y)) # fördimensionering
fig, ax = plt.subplots()
ax.plot(y)
k = 5
n = len(y)
for i in range(n): # loop över element
    n1 = np.max([0,i-k]) # n1 aldrig mindre än 0
    n2 = np.min([n,i+k+1]) # n2 aldrig större än n
    y_median[i] = np.median(y[n1:n2]) # medelvärde i fönster
ax.plot(y_median,'r')
ax.set_xlim([0,n])
ax.set_ylabel(r'T i $^o$C',fontsize=14)
ax.tick_params(labelsize=14)
#%% Uppgift 4, raket simulering
n = 100000      #antal simuleringar
mass = 5000 + 100 * np.random.randn(n)
v = 400 + 70 * np.random.randn(n)
W = 1/2 * mass * v**2           #Rörelseenergi
fig, ax = plt.subplots()
ax.hist(W, bins=50)
print(f'Medelvärde av W: {np.mean(W):.2e}') #inkl formatering
print(f'Standardavvikelse av W: {np.std(W):.2e}') #inkl formatering
#%% Uppgift 5
# Räta linjens ekvation: y = k*x + m
a = np.array([-2, -1])      #punkt a [x, y]
b = np.array([2, 4])        #punkt b [x, y]
k = (a[1]-b[1])/(a[0]-b[0]) # k = dy/dx, lutningen på linjen
m = a[1] - k*a[0]           # m = offset, vart på y-axeln linjen skär
x = np.arange(-3, 4)
y = k*x+m

fig, ax = plt.subplots()
ax.scatter(a[0],a[1], c='red')
ax.scatter(b[0], b[1], c='red')
ax.plot(x, y)
#%% Uppgift 6
år = np.arange(1954, 1971, 2)   #array med år 1954 till 1970 i steg om 2
antal = np.array([45, 33, 26, 16, 10, 8, 6, 6, 5])  #array med antal
r = 0.17
x = np.arange(1954, 1981, 2)
N = 45 * np.exp(-r*(x-1954))

fig, ax = plt.subplots()
ax.scatter(år, antal)
ax.plot(x, N, '#FFA500')   #plottar antal över år, med orange linje
ax.set_xlabel('År')
ax.set_ylabel('Antalet par')
ax.plot([1954, 1980], [1,1], 'r--')  #plottar en streckad linje => 1975