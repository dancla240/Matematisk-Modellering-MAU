# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:06:25 2024

Inlämning 1

@author: Daniel Claesson

Länk till youtube video: https://youtu.be/T2HQfLRKbMQ
"""

#%% Uppgift 1a
# summa.py
s = 0 # sätt summa till noll
for i in range(1,101): # loopa från 1 till 100, måste skriva 101 inte 100
    s = s + i # addera term till summan
print('Summan är ',s) # skriv ut resultatet

#%% Uppgift 1b
s = 0 # sätt summa till noll
for i in range(1,11): # loopa från 1 till 100, måste skriva 101 inte 100
    s = s + i # addera term till summan
print('Summan är ',s) # skriv ut resultatet
#%% Uppgift 2a
print(f"9**(1/2) = {9**(1/2)}") # ger 3
print(f"9**1/2 = {9**1/2}") # ger 4.5, dvs det är skillnad
#%% Uppgift 2b
print(round(1.52E-8 * 6.18E9,1)) # ger 93.9
#%% Uppgift 3a
y = lambda x : k*x + m
#%% Uppgift 3b
y = lambda x : 2*x + 3
y(1) # ger 5
#%% Uppgift 3b, alternativ lösning
y = lambda k, x, m : k*x + m
y(2, 1, 3) # ger 5
#%% Uppgift 4
u = [10, 20, 3, 4, -52] # skapar listan u med 5 element
#a)
print(len(u)) # längden på listan (5)
#b)
print(u[0]) # skriver ut första elementet i listan (10)
print(u[1]) # skriver ut andra elementet (20)
print(u[-1]) # skriver ut sista elementet (-52)
#c) sortera listan
u.sort()
print(f"Skriver ut listan efter sortering: {u}")
#%% Uppgift 5a)
Ugamla     = float(input('Ange U-värde för gamla fönster ')) #ange 
Unya       = float(input('Ange U-värde för nya fönster '))
area       = float(input('Ange Fönsterarea '))
gradtimmar = float(input('Ange Gradtimmar för orten '))
elpris     = float(input('Ange Elpris '))

energibesparing   = (Ugamla - Unya)*area*gradtimmar
kostnadsbesparing = energibesparing*elpris

print('Energibesparing  ', round(energibesparing), 'kWh per år')
print('Kostnadsbesparing', round(kostnadsbesparing), 'kr per år')
# Mitt svar:
# Energibesparing   7910 kWh per år
# Kostnadsbesparing 15820 kr per år
#%% Uppgift 5b)
fast_avgift = float(input('Fast avgift:'))
antal = int(input("Antal:"))
pris_per_styck = float(input('Styckpris:'))
pris_u_moms = fast_avgift + antal * pris_per_styck
pris_m_moms = pris_u_moms*1.25
print(f"Pris u moms: {int(pris_u_moms)} kr.")
print(f"pris m moms: {int(pris_m_moms)} kr.")
# svar med 10000 kr fast avg, 50 stycken och 2000 per styck:
# Pris_u_moms = 110000
# Pris_m_moms = 137500
#%% Uppgift 6a)
import numpy as np #importerar numpy
x = np.array([1,2,3])
y = np.array([4,5,6])
print(x,y) # ger [1 2 3] [4 5 6]
#%% Uppgfit 6b)
s = x + 5
print(s) # ger [6 7 8]
#%% Uppgift 6c)
t = x * 8
print(t) # ger [ 8 16 24]
#%% Uppgift 6d) Elementvis addition
u = x + y
print(u) # ger [5 7 9]
#%% Uppgift 6e) kvadratroten ur alla tal i vektorn x
v = np.sqrt(x)
print(v) # ger [1.         1.41421356 1.73205081]
#%% Uppgift 7a) Tillämpning: temperaturdata
import numpy as np

path = r'C:/Users/danie/OneDrive/Dokument/6 - Utbildning/S_2024_Python_Malmö_Universitet/Uppgift1/T10365.txt'
# Läs in temperaturvärden från filen T10365.txt.
# Lagra i 10 x 365 matrisen T
T = np.loadtxt(path)

# max, min, medelvärde för 1981-90, avrunda till en decimal
print()        # ger en tom rad
print('Maximal temperatur under 10 år', np.round( np.max(T),1 ) )
print('Minimal temperatur under 10 år', np.round( np.min(T),1 ) )
print('Medeltemperatur under 10 år   ', np.round( np.mean(T),1 ) )

# medeltemp. för jan. (dag 1-31) och jun. (dag 152-181), avrunda till en decimal
print()
print('Medeltemperatur januari ', np.round( np.mean(T[:,0:31]),1 ) )
print('Medeltemperatur juni    ', np.round( np.mean(T[:,151:181]),1 ) )
   
# årsmedelvärde för 1981 till 1990, avrunda till en decimal
print()
print('Årsmedeltemperatur för 1981 ', np.round( np.mean(T[0,:]),1 ) )
print('Årsmedeltemperatur för 1990 ', np.round( np.mean(T[9,:]),1)  )
#%% Uppgift 7b) Modifierar programmet i 7a
#i)
# Läs in temperaturvärden från filen T10365.txt.
# Lagra i 10 x 365 matrisen T
import numpy as np
path = r'C:/Users/danie/OneDrive/Dokument/6 - Utbildning/S_2024_Python_Malmö_Universitet/Uppgift1/T10365.txt'
T = np.loadtxt(path)
print(T.shape)

(m,n) = np.unravel_index(np.argmax(T), T.shape)
print(f"m = {m}") # ger 2
print(f"n = {n}") # ger 220
#%% ii) Beräknar medeltemp för alla rader och kolumnerna 32-59
np.mean(T[:,32:60]) # ger -0.7
#%% iii) Årsmedeltemp för 1985
np.mean(T[4,:]) # ger 7.6

