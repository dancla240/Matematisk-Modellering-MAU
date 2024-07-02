# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:32:11 2024

@author: danie

Namn: Daniel Claesson 

Youtube: https://youtu.be/Mbb1FdGZp1s
"""
#%% Uppgift 1 
import numpy as np 
import matplotlib.pyplot as plt 
 
x = np.linspace(0, 10) 

f = x / (1 + x) 
g = x**2 / (1 + x**2) 

fig, ax = plt.subplots() 
ax.plot(x, f, 'g--') 
ax.plot(x, g, 'r') 
ax.grid('on') 
ax.tick_params(labelsize = 14)

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
#%% Uppgift 1 med lite andra 'tick_params':
import numpy as np 
import matplotlib.pyplot as plt 
 
x = np.linspace(0, 10) 

f = x / (1 + x) 
g = x**2 / (1 + x**2) 

fig, ax = plt.subplots() 
ax.plot(x, f, 'g--') 
ax.plot(x, g, 'r') 
ax.grid('on') 
ax.tick_params(labelsize = 20,
               grid_linewidth=2,
               grid_color='g',
               grid_alpha=0.5,
               grid_linestyle='dashed')

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
#%%
help(plt.tick_params)
#%% Uppgift 2 
t = np.linspace(1790, 1960, 18) 
y = (3929000, 5308000, 7240000, 9638000, 12866000, 17069000, 23192000, 31443000, 38558000, 50156000, 62948000, 75995000, 91972000, 105710000, 122775000, 131669000, 150697000, 179323000) 
N = 197273000 / (1 + np.exp(-0.03134 * (t-1913.25)))

fig, ax = plt.subplots()
ax.set_title('Befolkningsutveckling i USA, 1790-1960', fontsize=10)
ax.set_xlabel('År')
ax.set_ylabel('Befolkningsstorlek')
ax.plot(t, y, 'o', t, N, '-') 
#%% Uppgift 3a
path = r'C:/Users/danie/OneDrive/Dokument/6 - Utbildning/S_2024_Python_Malmö_Universitet/Uppgift2/T10365-1.txt'
T = np.loadtxt(path)
T81 = T[0] # temperaturen för 1981 i en egen variabel T81
fig, ax = plt.subplots()
ax.hist(T81, bins=15)
ax.set_xlabel('temperatur år 1981')
ax.set_ylabel('frekvens')
#%% Uppgift 3b
T84 = T[4] # temperaturen för 1984 i en egen variabel T84
fig, ax = plt.subplots()
ax.hist(T84, bins=15)
ax.set_xlabel('temperatur år 1984')
ax.set_ylabel('frekvens')
#%%
help(plt.hist)
#%% Uppgift 4
tal_a = int(input('Ange tal a: '))
tal_b = int(input('Ange tal b: '))

if tal_b == 0:
    print('Division med noll ej tillåtet.')
else:
    kvot = tal_a / tal_b
    print(f'kvoten av tal a och tal b är {kvot:.2f}') # beräknar kvot och avrundar till 2 decimaler
#%% Uppgift 5a/b Läser in filen CCD.txt och plottar den.
path = r"C:/Users/danie/OneDrive/Dokument/6 - Utbildning/S_2024_Python_Malmö_Universitet/Uppgift2/CCD.txt"
C = np.loadtxt(path)
#5b: plottar bilden med gråskala och skalning 3 och 7
fig, ax = plt.subplots()
ax.imshow(C, cmap='gray', vmin=3, vmax=7)
#%% 5c, rättar till defekta pixlar:
i, j = np.shape(C)
for row in range(1, i-1):
    for col in range(1, j-1):
        if C[row, col] == 0:
            C[row, col] = 1/8 * (C[row-1, col-1] + C[row-1, col] + C[row-1, col+1] + C[row, col-1] + C[row, col+1] + C[row+1, col-1] + C[row+1, col] + C[row+1, col+1])
#%% 5d, plottar den justerade figuren
fig, ax = plt.subplots()
ax.imshow(C, cmap='gray', vmin=3, vmax=7)
#%% Tillämpning: Uppgift 6, ändra färger
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
A = plt.imread(r'C:/Users/danie/OneDrive/Dokument/6 - Utbildning/S_2024_Python_Malmö_Universitet/Uppgift2/havsorn_PO.jpg')  # läs bild och spara i A
ax.imshow(A)                      # bild
ax.axis('equal')
ax.axis('off')
#%% byter blått till rosa
fig,ax = plt.subplots()
A = plt.imread(r'C:/Users/danie/OneDrive/Dokument/6 - Utbildning/S_2024_Python_Malmö_Universitet/Uppgift2/havsorn_PO.jpg')  # läs bild och spara i A
A = A.astype(dtype='float')       # omvandla till float

RGB    = [1,122,195]              # blå färg som vi vill byta ut
RGB_ny = [255,192,203]            # ny färg, ljus rosa  

(m,n,o) = A.shape                 # dimensionerna på bilden, m antal rader, n djupet

for i in range(m):                # loopa över rader och kolonner
    for j in range(n):
        a = (A[i,j,0]-RGB[0])**2 + (A[i,j,1]-RGB[1])**2 + (A[i,j,2]-RGB[2])**2
        b = RGB[0]**2 + RGB[1]**2 + RGB[2]**2
        dist = np.sqrt(a/b)       # färgavstånd
        if dist < 0.1:            # om färgavstånd litet, byt ut färgen
            A[i,j,:] = RGB_ny

A = A.astype(dtype='int')         # måste omvandla till heltal innan vi visar
ax.imshow(A)                      # visa modifierad bild
ax.axis('equal')
ax.axis('off')
#%% byter allt som inte är blått till rosa
fig,ax = plt.subplots()
A = plt.imread(r'C:/Users/danie/OneDrive/Dokument/6 - Utbildning/S_2024_Python_Malmö_Universitet/Uppgift2/havsorn_PO.jpg')  # läs bild och spara i A
A = A.astype(dtype='float')       # omvandla till float

RGB    = [1,122,195]              # blå färg som vi vill byta ut
RGB_ny = [255,192,203]            # ny färg, ljus rosa  

(m,n,o) = A.shape                 # dimensionerna på bilden, m antal rader, n djupet

for i in range(m):                # loopa över rader och kolonner
    for j in range(n):
        a = (A[i,j,0]-RGB[0])**2 + (A[i,j,1]-RGB[1])**2 + (A[i,j,2]-RGB[2])**2
        b = RGB[0]**2 + RGB[1]**2 + RGB[2]**2
        dist = np.sqrt(a/b)       # färgavstånd
        if dist > 0.1:            # om färgavstånd litet, byt ut färgen
            A[i,j,:] = RGB_ny

A = A.astype(dtype='int')         # måste omvandla till heltal innan vi visar
ax.imshow(A)                      # visa modifierad bild
ax.axis('equal')
ax.axis('off')




