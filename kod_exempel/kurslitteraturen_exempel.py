# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:44:19 2024

@author: Per
"""

###############################################################################
#                                                                             #
#     Alla exempel ges i celler. Ställ dig i cellen och tryck                 #
#     på ikonen sidan om den gröna pilen i vertygsfältet, se kapitel 1.5      #
#     Alla filer som behövs är inkluderade i zip-filen                        #
#                                                                             #
###############################################################################

#%%
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX       XXXX                                 #
#     XX XX       XX XX      XX  XX         XX                                #
#     XXX        XX   XX     XXXXX        XX                                  #
#     XX xX     XX XX  XX    XX          XX                                   #
#     XX  XX   XX       XX   XX         XXXXXX                                #
#                                                                             #
###############################################################################

#%% Exempel 2.2
import numpy as np
r = 3
print(np.pi*r**2)   

#%% Exempel 2.3
import numpy as np
x = [1,2,4,2,7,8,2,13]
print(np.mean(x))

#%% Exempel 2.4
import numpy as np     # vi måste importera numpy för att använda roten
W = 50
m = 3
v = np.sqrt(2*W/m)
print('Farten är ',v)  # Skriv ut resultatet

#%% Exempel 2.5
s = 0                  # Sätt summa till noll
for i in range(1,101): # Loopa från 1 till 100, måste skriva 101 inte 100
    s = s + i          # Addera term till summan
 
print('Summan är ',s)  # Skriv ut resultatet

#%% Exempel 2.8a
print(12 + 13)    # heltal + heltal = heltal

#%% Exempel 2.8b
print(12.0 + 13)  # flyttal + heltal = flyttal

#%% Exempel 2.8c
print(2*3**2)     # först upphöjt till, prio 1, sedan multiplikation, prio 2)

#%% Exempel 2.8d
print(1/2*3)      # operationer från vänster till höger)

#%% Exempel 2.8e
print(int(4.6789))       # avhuggning

#%% Exempel 2.8f
print(round(4.6789,2))   # avrundning till två decimaler

#%% Exempel 2.8g
print(round(4.6789))     # avrundning till närmaste heltal

#%% Exempel 2.9a
print(11//4)      # kvot vid heltalsdivision

#%% Exempel 2.9b
print(11%4)       # rest vid heltalsdivision)

#%% Exempel 2.10
import numpy as np
x = 10
print(np.sqrt(x))

#%% Exempel 2.11
import numpy as np
print(np.round(4*np.pi,3))     # round(4*np.pi,3) går också bra

#%% Exempel 2.12
import numpy as np
print(np.fabs(25/8 - np.pi))     #  abs(25/8 - np.pi) går också bra)

#%% Exempel 2.13a
import numpy as np
f = lambda x: np.sqrt(x) - 1
print(f(4))

#%% Exempel 2.13b
import numpy as np
g = lambda x, k: k*x**2 
print(g(3,-1))

#%%
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX       XXXX                                 #
#     XX XX       XX XX      XX  XX         XX                                #
#     XXX        XX   XX     XXXXX         XX                                 #
#     XX xX     XX XX  XX    XX             XX                                #
#     XX  XX   XX       XX   XX         XXXXX                                 #
#                                                                             #
###############################################################################

#%% Exempel 3.2
m = 6
v = 3.0
W = 0.5*m*v**2
print(W)
print(type(W))

#%% Exempel 3.3
import numpy as np
x = 3
t = 3*x + 1                  # täljaren för sig och
n = np.sqrt(x**2 - 1) - 1    # nämnaren för sig
y = t/n                      # sätt ihop 
print(y)

####### Samma men mindre tydligt #####################

x = 3
y = (3*x + 1)/(np.sqrt(x**2 - 1) - 1)    
print(y)

#%% Exempel 3.5
u = [-1,3,4,2,-12,8]

print(len(u))

print(u[0])     # första elementet, index 0
print(u[3])     # fjärde elementet, index 3
print(u[-1])    # sista elementet, index -1
print(u[-2])    # näst sista elementet, index -2

u[1] = 52  
print(u)

#%% Exempel 3.7
name   = ['Gudrun', 'Jönsson']
age    = 24
gender = 'female' 
person = [name, age, gender]

print(person[0])
print(person[0][0])

#%% Exempel 3.8
u = [-3, -2, 0, 1, 4, 7]
 
#%% Exempel 3.8a
v = u[2:5]  # delområdet har 5 - 2 = 3 element 
print(v)

#%% Exempel 3.8b
v = u[1:]   # samma som u[1:len(u)]
print(v)
v = u[1:len(u)]
print(v)

#%% Exempel 3.8c
v = u[0:5:2]
print(v)

#%% Exempel 3.9
s = 'Hejsan hoppsan'
t = s[7:]
print(t)

#%% Exempel 3.10
u = [1,2,5,'hej',9]

#%% Exempel 3.10a
u.append(42)
print(u)

#%% Exempel 3.10b
u.insert(1,'Nisse')
print(u)

#%% Exempel 3.10c
u.remove(2)
print(u)

#%% Exempel 3.10d
del u[0]
print(u)

#%% Exempel 3.10e
del u[2:]
print(u)

#%% Exempel 3.11
x = [1,2,5,-4,2,-8]

#%% Exempel 3.11a
x.sort()       # x sorteras på plats
print(x)

#%% Exempel 3.11b
x.reverse()    # vänder på ordningen av x på plats
print(x)

#%% Exempel 3.12a
s = 'Massan av partikeln'
ord = s.split()
print(ord)

#%% Exempel 3.12b
s = 'massa,hastighet'
ord = s.split(',')
print(ord)

#%% Exempel 3.13a
l1 = ['Lösningen','till','ekvationen']
l2 = ['massa','impuls','rörelsemängd']
s1 = ' '.join(l1)   # sätt ihop med ett blanktecken
s2 = ', '.join(l2)  # sätt ihop med komma och ett blanktecken
print(s1)
print(s2)

#%% Exempel 3.13b
m = 5.8
l3 = ['massan är',str(m),'kilogram']
s3 = ' '.join(l3)   # sätt ihop med ett blanktecken
print(s3)

#%% Exempel 3.14
s = '  farten är  '
   
#%% Exempel 3.14a
s1 = s.strip()
print(s1)

#%% Exempel 3.14b
s2 = s.lstrip()
print(s2)

#%% Exempel 3.14c
s3 = s2.replace('farten','hastigheten')
print(s3)

#%%                    
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX        XX  XX                              #
#     XX XX       XX XX      XX  XX       XX  XX                              #
#     XXX        XX   XX     XXXXX        XXXXXX                              #
#     XX xX     XX XX  XX    XX               XX                              #
#     XX  XX   XX       XX   XX               XX                              #
#                                                                             #
###############################################################################

#%% Exempel 4.2
m = float(input('Ge m '))  # omvandla input till flyttal
v = float(input('Ge v '))  # omvandla input till flyttal
W = 0.5*m*v**2
print('Kinetisk energi',W)

#%% Exempel 4.3
b = 3
h = 5
A = b*h/2
print('Om basen är',b, 'och höjden är',h,'blir arean',A)

#%% Tillämpning: fönsterbyte
Ugamla     = float(input('U-värde för gamla fönster '))
Unya       = float(input('U-värde för nya fönster '))
area       = float(input('Fönsterarea '))
gradtimmar = float(input('Gradtimmar för orten '))
elpris     = float(input('Elpris '))

energibesparing   = (Ugamla - Unya)*area*gradtimmar
kostnadsbesparing = energibesparing*elpris

print('Energibesparing  ', round(energibesparing), 'kWh per år')
print('Kostnadsbesparing', round(kostnadsbesparing), 'kr per år')

#%%                                       
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX         XXXXX                              #
#     XX XX       XX XX      XX  XX       XX                                  #
#     XXX        XX   XX     XXXXX        XXXXXX                              #
#     XX xX     XX XX  XX    XX               XX                              #
#     XX  XX   XX       XX   XX           XXXXX                               #
#                                                                             #
###############################################################################

#%% Exempel 5.1
import numpy as np
x = np.array([5,-4,-6])
print(x)
print(x.dtype)
print(x.size)
x[1] = 92
print(x)
x[0] = 10.6 # talet trunkeras, huggs av, innan tilldelningen
print(x)
x = x.astype(dtype='float')
print(x)

#%% Exempel 5.2
import numpy as np
x = np.array([5.0,-4.0,-6.0])
print(x)
print(x.dtype)
x[0] = 10.6 
print(x)

#%% Exempel 5.3
import numpy as np
x = np.arange(20,24)         # resultatet blir heltal
print(x)

#%%
x = np.arange(20.0,24.0)     # resultatet blir flyttal
print(x)

#%%
x = np.arange(20,24,dtype='float')  # resultatet blir flyttal
print(x)

#%%
x = np.arange(20.0,22.2,0.3)
print(x)

#%%
x = np.arange(10.0,0.0,-2.0)
print(x)

#%%
x = np.linspace(0,10)
print(x)

#%%
x = np.linspace(0,10,100)
print(x)

#%% Exempel 5.4
import numpy as np
x = np.array([-3,-2,0,1,4,7])

#%% Exempel 5.4a
y = x[2:5]        # delområdet har 5 - 2 = 3 element
print(y)

#%% Exempel 5.4b
y = x[2:]         # från index 2 till sista 
print(y)

#%% Exempel 5.5
import numpy as np
v = np.array([5,2,3,4,9,7])
print(v)

#%% Exempel 5.5a
v[0:5:2] = [0,6,8]
print(v) 

#%% Exempel 5.5b
v[:3] = [7,8,9]     
print(v)

#%% Exempel 5.6
import numpy as np
A = np.array([[1,-2,3],
              [0,5,4]])
print(A)

print(A.dtype)      # typen

print(A.ndim)       # antal index

print(A.shape)      # antal rader och kolonner

(m,n) = A.shape
print(m,n)

#%% Exempel 5.7
import numpy as np
A = np.zeros((2,3))  # även np.zeros([2,3]) går bra
print(A)

#%%
A = np.ones((3,3))
print(A)

#%% Exempel 5.8
import numpy as np
v = np.random.rand(5)         # vektor med 5 likformigt fördelade slumptal
print(v)

#%%

A = np.random.randn(4,3)      # 4 x 3 matris med normalfördelade slumptal
print(A)

#%%

v = np.random.randint(1,7,15) # 15 slummässiga heltal mellan 1 och 6, 
print(v)

#%% Exempel 5.9
import numpy as np
A = np.array([[1,2,3,4],      # rad index 0
              [5,6,7,8],      # rad index 1
              [9,10,11,12]])  # rad index 2
print(A)

#%% Exempel 5.9a
B = A[1:3,1:4]
print(B)

#%% Exempel 5.9b
B = A[-1,1:4]  
print(B)

#%% Exempel 5.9c
C = A[2,:]
print(C)

#%% Exempel 5.9d
D = A[:,1]  # omvandlar automatiskt till radvektor
print(D)

#%% Exempel 5.10a
import numpy as np
a = np.array([[1,2],
              [3,4],
              [5,6]])
np.savetxt('data1.txt',a)

#%% Exempel 5.10b
import numpy as np
b = np.loadtxt('data1.txt')
print(b)

#%% Exempel 5.10c
np.savetxt('data2.txt',a, delimiter=',')

#%% Exempel 5.10d
b = np.loadtxt('data2.txt', delimiter=',')
print(b)

#%% Exempel 5.10e
v = np.loadtxt('data2.txt', delimiter=',', usecols=(1))
print(v)

#%% Exempel 5.11
import numpy as np
x = np.array([1,2,3])

#%% Exempel 5.11a
np.save('data1', x)

#%% Exempel 5.11b
y = np.load('data1.npy')
print(y)

#%% Exempel 5.12
import numpy as np
x = np.array([1,2,3])
a = np.array([[0,4],
              [7,6]])
print(x)
print(a)

#%% Exempel 5.12a
with open('data2.npy', 'wb') as f:  # wb = write binary
    np.save(f, x)     # indentera (dra in) 4 positioner
    np.save(f, a)  
    
#%% Exempel 5.12b    
with open('data2.npy', 'rb') as f:  # rb = read binary
    x = np.load(f)    # indentera (dra in) 4 positioner
    a = np.load(f)	   
print(x)
print(a)

#%% Exempel 5.12c
np.savez('data3',x=x, a=a)

#%% Exempel 5.12d
npzfile = np.load('data3.npz')
print(npzfile.files)

x = npzfile['x']
a = npzfile['a']
print(x)
print(a)

#%% Exempel 5.13
import numpy as np
x = np.array([1,2,6])      
y = np.array([-1,1,2])  

A = np.array([[1,1],
              [4,6]])    
B = np.array([[4,1],
              [2,3]])

#%% Exempel 5.13a
z = 3*x
print(z)

#%% Exempel 5.13b
u = x + 2   # exempel på så kallad broadcasting
print(u)

#%% Exempel 5.13c
v = x*y
print(v)

#%% Exempel 5.13d
w = x**2
print(w)

#%% Exempel 5.13e
C = A + B
print(C)

#%% Exempel 5.13f
D = A/B
print(D)

#%% Exempel 5.13g
E = A*B      # elementvis multiplikation
print(E)

#%% Exempel 5.14
import numpy as np
x = np.array([1, 4, 9, 16])    
A = np.array([[1.1, 1.9],
              [2.5, 3.6]])  

#%% Exempel 5.14a
y = np.sqrt(x)
print(y)

#%% Exempel 5.14b
B = np.round(A)
print(B)

#%% Exempel 5.15
import numpy as np
A = np.array([[1,5],
              [4,0],
              [2,8]])

#%% Exempel 5.15a
B = np.sort(A,axis=1)
print(B)

#%% Exempel 5.15b
B = np.sort(A,axis=0)
print(B)

#%% Exempel 5.15c
print(np.sum(A))  

#%% Exempel 5.15d
s0 = np.sum(A,axis=0)   
print(s0)

#%% Exempel 5.15e
s1 = np.sum(A,axis=1)   
print(s1)

#%% Exempel 5.15f
print(np.max(A))
print(np.argmax(A))
(m,n) = np.unravel_index(np.argmax(A),A.shape)
print(m,n)

#%% Exempel 5.15g
max0 = np.max(A,axis=0)   
print(max0)

#%% Exempel 5.15h
argmax0 = np.argmax(A,axis=0)   
print(argmax0)

#%% Exempel 5.16
import numpy as np
A = np.array([[1,2,3,0],
              [8,6,4,5],
              [9,8,3,1]])

#%% Exempel 5.16a
print(np.mean(A[0:2,1:4]))

#%% Exempel 5.16b
print(np.median(A[:,3]))
print(np.median(A[:,-1]))

#%% Tillämpning: temperaturdata
import numpy as np

# Läs in temperaturvärden från filen T10365.txt.
# Lagra i 10 x 365 matrisen T
T = np.loadtxt('T10365.txt')
	
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

#%%                                       
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX         XXXXXX                             #
#     XX XX       XX XX      XX  XX       XX                                  #
#     XXX        XX   XX     XXXXX        XX XXXX                             #
#     XX xX     XX XX  XX    XX           XX    XX                            #
#     XX  XX   XX       XX   XX            XXXXXX                             #
#                                                                             #
###############################################################################

#%% Exempel 6.1
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,4)          # vektor med x-värden
y = x**2                      # vektor med y-värden
fig, ax = plt.subplots()      # skapa instanserna fig och ax
ax.plot(x,y)                  # plotta y mot x
ax.tick_params(labelsize=14)  # sätt storlek på griddmarkeringarna

#%% Exempel 6.2
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-5,5,10)      # för få x, kantig plot
y = x**2
fig, ax = plt.subplots()
ax.plot(x, y)
ax.tick_params(labelsize=14)

#%% Exempel 6.3a
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,5)
y1 = x**2
y2 = 2*x - 3
fig, ax = plt.subplots()
ax.plot(x,y1,'+',x,y2,'o')
ax.tick_params(labelsize=14)

#%% Exempel 6.3b
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3,3)
y1 = 2 + 0*x    # vektor med bara tvåor
y2 = x**2
y3 = 2*x + 2
fig, ax = plt.subplots()
ax.plot(x,y1,':')             # prickad
ax.plot(x,y2,linewidth=2)
ax.plot(x,y3,linewidth=3) 
ax.tick_params(labelsize=14)     

#%% Exempel 6.4
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(2,2)   # ax blir 2 x 2 matris
x = np.linspace(-2,2)
y0 =  x
y1 = -x
y2 =  x**2
y3 = -x**2
ax[0,0].plot(x,y0)            # uppe till vänster, index (0,0)
ax[0,1].plot(x,y1)            # uppe till höger, index (0,1)
ax[1,0].plot(x,y2)            # nere till vänster, index (1,0)
ax[1,1].plot(x,y3)            # nere till höger, index (1,1)

#%% Exempel 6.5
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-10,10,500)
y = 1/x
fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
ax.tick_params(labelsize=14)

# dela upp i två

x1 = np.linspace(-10,-1.e-10,500)
y1 = 1/x1
x2 = np.linspace(1.e-10,10,500)
y2 = 1/x2
fig, ax = plt.subplots()
ax.plot(x1,y1,'b',x2,y2,'b')
ax.axis([-5,5,-15,15])    
ax.tick_params(labelsize=14)

#%% Exempel 6.6
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1,1,200)      # tag många x-värden för jämn plot
y = np.sqrt(1 - x**2)

fig, ax = plt.subplots()
ax.plot(x,y,'b')               # övre delen av cirkeln
ax.plot(x,-y,'b')              # undre delen av cirkeln
ax.tick_params(labelsize=14)

# plot med samma skaldelar på axlarna
x = np.linspace(-1,1,200)
y = np.sqrt(1 - x**2)

fig, ax = plt.subplots()
ax.plot(x,y,'b')               # övre delen av cirkeln
ax.plot(x,-y,'b')              # undre delen av cirkeln
ax.tick_params(labelsize=14)
ax.axis('equal')

#%% Exempel 6.7a
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,2)
y1 = x**2
y2 = np.sqrt(x)
fig, ax = plt.subplots()                 
ax.plot(x,y1,label=r'$y=x^{2}$')             # r gör att sträng tolkas som
ax.plot(x,y2,'--',label=r'$y=\sqrt{x}$')     # LaTeX-kod
ax.set_title('Potensfunktioner',fontsize=14)
ax.set_xlabel('x',fontsize=14)
ax.set_ylabel('y',fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)

#%% Exempel 6.7b
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0,4)            # generera t-värden
u1 = t/(1 + t)                  # generera u = t/(1 + t)
u2 = t**3/(1 + t**3)            # generera u = t^3/(1 + t^3)
fig, ax = plt.subplots()
ax.plot(t,u1,'+')               # plotta (plus)
ax.plot(t,u2,'o')               # plotta (cirklar)
ax.set_title('Rationella funktioner',fontsize=14)
ax.text(2.9,0.6,r'$\frac{t}{1 + t}$',fontsize=16) 
ax.text(0.8,0.8,r'$\frac{t^3}{1 + t^3}$',fontsize=16)
ax.set_xlabel('t',fontsize=14)  # skriv text på x-axeln
ax.set_ylabel('u',fontsize=14)  # skriv text på y-axeln
ax.tick_params(labelsize=14)

#%% Exempel 6.8a
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,2)
y1 = 1/(x**2 + 1)
y2 = 1/(x**4 +1)
fig, ax = plt.subplots()
ax.plot(x,y1,label=r'$\alpha = 2$')
ax.plot(x,y2,'--',label=r'$\alpha = 4$')
ax.legend(fontsize=14)

# även om lång rad får man inte bryta den för då blir
# LaTeX typsättningen fel
ax.set_title(r'Funktionen $y = \frac{1}{x^{\alpha} + 1},0 \leq x \leq 2$',fontsize=14)
ax.tick_params(labelsize=14)

#%% Exempel 6.8b
import matplotlib.pyplot as plt
import numpy as np
fix, ax = plt.subplots()
ax.plot([0,10,10,0,0],[0,0,10,10,0]) # rita fyrkant
ax.text(3,1,'Python',fontsize=10)
ax.text(3,3,'Python',fontsize=20)
ax.text(3,5,'Python',fontsize=30)
ax.text(3,7,'Python',fontsize=40)
ax.axis('off')

#%% Exempel 6.9
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-5,5)
y1 = -2*x + 5 
y2 = x**2
fig, ax = plt.subplots()
ax.plot(x,y1)
ax.plot(x,y2)
ax.tick_params(labelsize=14)
ax.annotate(r'$-2x + 5$',fontsize=14, xy=(-1,7),\
  xytext=(1,15),arrowprops=dict(width=1))
ax.annotate(r'$x^2$',fontsize=14, xy=(-4,16),\
  xytext=(-2,19),arrowprops=dict(width=1))
    
#%% Exempel 6.10
import matplotlib.pyplot as plt
import numpy as np

T = np.loadtxt('T.txt')

fig, ax = plt.subplots()
ax.hist(T,5,edgecolor='black')
ax.tick_params(labelsize=14)

fig, ax = plt.subplots()
m, bins, patches = ax.hist(T,bins=5,edgecolor='black')
ax.tick_params(labelsize=14) 
print(m)      # antalet element (frekvensen) i varje klass
print(bins)   # klassgränser

fig, ax = plt.subplots()
ax.hist(T,20,edgecolor='black')   # 20 klasser
ax.tick_params(labelsize=14)
 
#%% Exempel 6.11
import matplotlib.pyplot as plt
import numpy as np
# likformigt fördelade mellan 0 och 1
x = np.random.rand(1000000)   
fig, ax = plt.subplots()
ax.hist(x,10,edgecolor='black')
ax.tick_params(labelsize=14)

# normalfördelade, medelvärde 0, standardavvikelse 1
x = np.random.randn(1000000)  
fig, ax = plt.subplots()
ax.hist(x,50,edgecolor='black')
ax.tick_params(labelsize=14)

#%% Exempel 6.12
import matplotlib.pyplot as plt
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,-1,0.5,2,3,5,-2,-4,4,9])

fig, ax = plt.subplots()
ax.bar(x,y)
ax.tick_params(labelsize=14)

fig, ax = plt.subplots()
ax.stem(x,y)
ax.tick_params(labelsize=14)

#%% Exempel 6.13
import matplotlib.pyplot as plt
import numpy as np
A = np.array([[44, 50, 52, 50, 44, 36, 27, 19, 12, 7 ], \
              [58, 66, 69, 66, 58, 48, 36, 25, 16, 9 ], \
              [71, 81, 84, 81, 71, 58, 44, 31, 19, 11], \
              [81, 91, 95, 91, 81, 66, 50, 34, 22, 13], \
              [84, 95, 99, 95, 84, 69, 52, 36, 23, 13], \
              [81, 91, 95, 91, 81, 66, 50, 34, 22, 13], \
              [71, 81, 84, 81, 71, 58, 44, 31, 19, 11], \
              [58, 66, 69, 66, 58, 48, 36, 25, 16, 9]])
    
#%% Exempel 6.13a
fig,ax = plt.subplots()
pos = ax.imshow(A,cmap='jet')
fig.colorbar(pos)
ax.tick_params(labelsize=14)

#%% Exempel 6.13b
fig,ax = plt.subplots()
pos = ax.imshow(A,vmin=40,vmax=90,cmap='jet')
fig.colorbar(pos)
ax.tick_params(labelsize=14)

#%% Exempel 6.14
import matplotlib.pyplot as plt
A = plt.imread('Lions.jpg')
print(A.shape)
print(A.dtype)
fig, ax = plt.subplots()
ax.imshow(A)
ax.axis('equal')        # samma skala på axlarna
ax.axis('off')          # ta bort axlar


#%% Exempel 6.15
import matplotlib.pyplot as plt
A = plt.imread('Lions.jpg')
A = A.astype(dtype='float')             # obs, omvandla från uint8 till flyttal
B = (A[:,:,0] + A[:,:,1] + A[:,:,2])/3  # medelvärdesbilda
fig, ax = plt.subplots()
ax.imshow(B,cmap='gray')                # plotta matrisen med färgskalan gray
ax.axis('equal')                        # samma skala på axlarna
ax.axis('off')                          # ta bort axlarna

#%% Exempel 6.16
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-2,2)
y = 2*x - 1
# normal python plot
fig, ax = plt.subplots()
ax.plot(x,y)
ax.tick_params(labelsize=14)
ax.grid('on')

# plot med två-axlar genom origo
fig, ax = plt.subplots()
ax.plot(x,y)
ax.tick_params(labelsize=14)
ax.grid('on')

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

#%%                                       
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX         XXXXXX                             #
#     XX XX       XX XX      XX  XX            XX                             #
#     XXX        XX   XX     XXXXX            XX                              #
#     XX xX     XX XX  XX    XX              XX                               #
#     XX  XX   XX       XX   XX             XX                                #
#                                                                             #
###############################################################################

#%% Exempel 7.3a
n = int(input('Ge ett heltal '))
if n%2 == 0:  
    print('Heltalet är jämnt')
else:       
    print('Heltalet är udda ')
    
#%% Exempel 7.3.b
x = float(input('Ge ett positivt tal '))
if x < 0:  
    print('Talet är inte positivt')
elif 0 <= x <= 10:       
    print('Talet ligger mellan 0 och 10 ')
else:
    print('Talet är större än 10')
    
#### eller #####

x = float(input('Ge ett positivt tal '))
if x < 0:  
    print('Talet är inte positivt')
elif 0 <= x <= 10:       
    print('Talet ligger mellan 0 och 10 ')
elif x >10:
    print('Talet är större än 10')
    
#%% Exempel 7.4
import numpy as np
p = float(input('Ge p '))
q = float(input('Ge q '))
# Beräkna diskriminanten
d = (p/2)**2 - q
if d > 0:
    print('Två lösningar: ', -p/2 + np.sqrt(d), -p/2 - np.sqrt(d))
elif d == 0:
    print('En lösning: ', -p/2)
else:
    print('Saknar reella lösningar')

#%% Exempel 7.5
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# rita tavla
ax.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1])

# rita stor och liten cirkel
x = np.linspace(-1,1,100)
y = np.sqrt(1-x**2)
ax.plot(x,y,'k')
ax.plot(x,-y,'k')
ax.plot(0.1*x,0.1*y,'r')
ax.plot(0.1*x,-0.1*y,'r')

# samma skala på x- och y-axeln
# ta bort axelmarkeringarna
ax.axis('equal')
ax.axis('off')

hit = 0    # initialisera
n   = 0    # initialisera
while hit < 1:                      # fortsätt så länge ingen träff
    x = -1 + 2*np.random.rand()     # slumpad x-koord. mellan -1 och 1
    y = -1 + 2*np.random.rand()     # slumpad y-koord. mellan -1 och 1
    ax.plot(x,y,'kx')               # plotta var kastet hamnar
    n = n + 1                       # antal kast ökar med 1
    if np.sqrt(x**2 + y**2) < 0.1:  # träff - ändra värdet på hit
        hit = 1        
print('Antal kast ',n)              # utanför loopen, inget indrag

#%% Exempel 7.6
v = 0                           # volymen 0 då vi börjar (initialisering)
n = 0                           # antalet spannar 0 då vi börjar 
while v < 4990:                 # iterera så länge v < 4990
    v = v + 25                  # fyller i en spann, volymen ökar med 25
    n = n + 1                   # antalet ifyllda spannar ökar med 1
print('Antal spannar ',n - 1)   # utanför loopen, inget indrag
print('Volym vatten  ',v - 25)

#%% Exempel 7.7
A   = float(input(' Ge ett värde på A '))
x0  = float(input(' Ge ett startvärde för roten '))
tol = float(input(' Ge toleransen '))
n = 0                     # initialisering 
dx = tol + 1              # se till att vi går in i loopen 
while dx > tol:           # iterera så länge dx > tol
    x1 = 0.5*(x0 + A/x0)  # bättre värde
    dx = abs(x1 - x0)     # avståndet x0 och x1  
    x0 = x1               # x1 nytt startvärde, dvs. x0 = x1
    n = n + 1             # stega upp iterationsräknaren
print('Roten är ',x0)     # utskrift
print('Iterationer ',n)   # utskrift

#%% Exempel 7.8
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# rita tavla
ax.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1])

# rita stor och liten cirkel
x = np.linspace(-1,1,100)
y = np.sqrt(1-x**2)
ax.plot(x,y,'k')
ax.plot(x,-y,'k')
ax.plot(0.1*x,0.1*y,'r')
ax.plot(0.1*x,-0.1*y,'r')

# samma skala på x- och y-axeln
# ta bort axelmarkeringarna
ax.axis('equal')
ax.axis('off')

n   = 0                             # initialisera
for i in range(200):                # loopa 200 gånger
    x = -1 + 2*np.random.rand()     # slumpad x-koord. mellan -1 och 1
    y = -1 + 2*np.random.rand()     # slumpad y-koord. mellan -1 och 1
    ax.plot(x,y,'kx')               # plotta var kastet hamnar
    if np.sqrt(x**2 + y**2) < 0.1:  # träff - ökar n med 1
        n = n + 1        
print('Antal träffar ',n)            # utanför loopen, inget indrag

#%% Exempel 7.9
import numpy as np
T = np.loadtxt('T.txt')  # läs in och lagra i T 
Tsum = 0                 # initialisera summan
for i in range(len(T)):
    Tsum = Tsum + T[i]   # addera term till summan
Tmedel = Tsum/len(T)     # beräkna medeltemperaturen
print('Medeltemperatur: ',  round(Tmedel,2))

# Kontroll
print('Medeltemperatur: ',  round(np.mean(T),2))

#%% Exempel 7.10
import numpy as np
A = np.zeros((3,6))         # fördimensionering
s = 0                       # initialisering
for i in range(3):          # loop över rader
    for j in range(6):      # loop över kolonner
        A[i,j] = 1/(i+j+1)  # generera element
        s = s + A[i,j]      # addera element till summan
print('Summan av elementen är',s)

# Kontroll
print('Summan av elementen är',np.sum(A))

#%% Exempel 7.11
import numpy as np
# radkolsum.py
A = np.array([[1,5],
              [4,0],
              [2,8]])     # 3 x 2 matris
print('Radsummor')
for i in range(3):        # loop över rader    
    s = 0                 # initialisera
    for j in range(2):    # loop över kolonner
        s = s + A[i,j]    # addera element till summan
    print('Summan av elementen i rad',i,'är',s)

print('Kolonnsummor')
for j in range(2):        # loop över kolonner
    s = 0                 # initialisera
    for i in range(3):    # loop över rader
        s = s + A[i,j]    # addera element till summan
    print('Summan av elementen i kolonn',j,'är',s)
    
#%% Tillämpning: signalbehandling
import numpy as np
import matplotlib.pyplot as plt
T = np.loadtxt('T8183.txt')
Tm = np.zeros(len(T))               # fördimensionering
fig, ax = plt.subplots()
ax.plot(T)
k = int(input('Ge värde på k '))
n = len(T)
for i in range(n):                  # loop över element
    n1 = np.max([0,i-k])            # n1 aldrig mindre än 0
    n2 = np.min([n,i+k+1])          # n2 aldrig större än n
    Tm[i] = np.mean(T[n1:n2])       # medelvärde i fönster
ax.plot(Tm,'r')
ax.set_xlim([0,n])
ax.set_ylabel(r'T i $^o$C',fontsize=14)
ax.tick_params(labelsize=14)

#%% Tillämpning: bildbehandling
import numpy as np
import matplotlib.pyplot as plt
B = plt.imread('PARISM.JPG')          # läs in
fig, ax = plt.subplots()      
ax.imshow(B,cmap='gray')              # plotta original
ax.axis('equal')
ax.axis('off')
(m,n) = B.shape                       # rader och kolonner

Bm = np.zeros((m,n))                  # fördimensionera 
for i in range(1,m-1):                # loop över rader
    for j in range(1,n-1):            # loop över kolonner
        Bm[i,j] = np.mean(B[i-1:i+2,j-1:j+2])  # medelvärde av grannar
fig, ax = plt.subplots()
ax.imshow(Bm[1:m-1,1:n-1],cmap='gray') # plotta
ax.axis('equal')
ax.axis('off')

D = B - Bm                            # skillnadsbild
fig, ax = plt.subplots()
ax.imshow(D[1:m-1,1:n-1],cmap='gray')  
ax.axis('equal')
ax.axis('off')

E = B + D                             # kantförstärkt bild
fig, ax = plt.subplots()
ax.imshow(E[1:m-1,1:n-1],cmap='gray') 
ax.axis('equal')
ax.axis('off')

#%% Tillämpning: ändra färger
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
A = plt.imread('havsorn_PO.jpg')  # läs bild och spara i A
ax.imshow(A)                      # bild
ax.axis('equal')
ax.axis('off')

fig,ax = plt.subplots()
A = plt.imread('havsorn_PO.jpg')  # läs bild och spara i A
A = A.astype(dtype='float')       # omvandla till float

RGB    = [1,122,195]              # blå färg som vi vill byta ut
RGB_ny = [255,192,203]            # ny färg, ljus rosa  

(m,n,o) = A.shape                 # dimensionerna på bilden, m antal rader
                                  # n djupet
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

#%% Tillämpning: avskogning

import numpy as np
import matplotlib.pyplot as plt

A = plt.imread('amazon_deforestation_20110805.jpg')  # läs in
fig, ax = plt.subplots()      
ax.imshow(A)                      # plotta originalbild

A = A.astype(dtype='float')       # omvandla till flyttal

RGB = np.zeros((3,3))             # fördimensionera
RGB[0,:] = [35,56,37]             # typiskt RGB för regnskog
RGB[1,:] = [157,126,98]           # typiskt RGB för avverkning 
RGB[2,:] = [216,221,243]          # typiskt RGB för moln

RGBNY = np.zeros((3,3))           # fördimensionera
RGBNY[0,:] = [0,255,0]            # klargrön, ny färg för regnskog
RGBNY[1,:] = [100,50,0]           # mörkbrun, ny färg för avverkning
RGBNY[2,:] = [255,255,255]        # kritvit, ny färg för moln

antal = np.zeros(3)               # vektor med tre element där vi lagrar
                                  # antalet pixlar i de olika klasserna
dist = np.zeros(3)                # vektor med tre element där vi lagrar
                                  # färgavståndet till de tre klasserna
(m,n,o) = A.shape                 # dimensionerna på bilden, m antal rader
                                  # n antalet kolonner, o djup                        
for i in range(m):                # loopa över rader och kolonner
    print('rad ',i,'av',m)        # för att se hur långt vi kommit        
    for j in range(n):
        for k in range(3):        # loopa över de tre klasserna 
            a = (A[i,j,0]-RGB[k,0])**2 + (A[i,j,1]-RGB[k,1])**2 + \
                (A[i,j,2]-RGB[k,2])**2
            b = RGB[k,0]**2 + RGB[k,1]**2 + RGB[k,2]**2
            dist[k] = np.sqrt(a/b)       # färgavstånd
         
        klass = np.argmin(dist)   # bestäm vilken klass som har minsta 
                                  # färgavstånd till pixeln
        antal[klass] = antal[klass] + 1  # öka antalet i denna klass med 1
        A[i,j,:] = RGBNY[klass,:] # omdefiniera färgen på pixeln i enlighet
                                  # med den bestämda klassen
A = A.astype(dtype='int')         # måste omvandla till heltal innan vi visar
ax.imshow(A)                      # visa modifierad bild
ax.axis('equal')
ax.axis('off')

print('regnskog   ',round(100*antal[0]/(m*n),1),'procent')
print('avverkning ',round(100*antal[1]/(m*n),1),'procent')
print('moln       ',round(100*antal[2]/(m*n),1),'procent')


#%%                                       
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX          XXXX                              #
#     XX XX       XX XX      XX  XX        XX  XX                             #
#     XXX        XX   XX     XXXXX          XXXX                              #
#     XX xX     XX XX  XX    XX            XX  XX                             #
#     XX  XX   XX       XX   XX             XXXX                              #
#                                                                             #
###############################################################################


#%% Exempel 8.3.

import numpy as np

def distance(x1,y1,x2,y2):
    """ 
    tar koordinaterna för två punkter (x1,y1) och (x2,y2)
    och beräknar avståndet m.h.a. Pythagoras sats 
    """
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return d

# här börjar det egentliga programmet
x1 = 1
y1 = 2
x2 = 3
y2 = -8

d = distance(x1,y1,x2,y2)   # anrop av funktionen
print('Avståndent är ',round(d,3))  

#%% Exempel 8.4.

def interp(x1,y1,x2,y2,x):
    """
    linjär interpolation mellan två punkter (x1,y1) and (x2,y2). 
    givet ett x beräknar vi motsvarande y
    """
    k = (y2-y1)/(x2-x1)   # riktningskoefficient
    m = y1 - k*x1         # m-värde
    y = k*x + m           # sätt in x i linjens ekvation
    return y

# här börjar det egentliga programmet
db1  = 285   # samhörande värden på drabrottsgäns
bhb1 = 85.5  # och Brinell HB-värden
db2  = 305
bhb2 = 90.2

# interpolera
db   = 290
bhb = interp(db1,bhb1,db2,bhb2,db)
print('Brinell HB-värde',round(bhb,1))  

#%% Exempel 8.5
import numpy as np
	
def altitude(T):
    h = ( G*M*T**2/(4*np.pi**2) )**(1/3) - R
    return h

# här börjar själva programmet
# definiera konstanter
G = 6.67e-11             # gravitationskonstanten
M = 5.97e24              # jordens massa
R = 6.371e6              # jordens radie

T = float(input('Ge omloppstid '))
h = altitude(T)/1000                  # h i km
print('Höjd i km',round(h))

#%% Exempel 8.6
import numpy as np
	
def mittensumma(f,a,b,n):
    """ area under grafen via mittensumma """
    dx = (b-a)/n
    xi = np.linspace(a+dx/2,b-dx/2,n)
    I = np.sum(f(xi))*dx
    return I

def f(x):
    y = x**2
    return y

# Här börjar själva programmet 
a, b = input('Ge a och b, kommaseparerade ').split(',')
a, b = float(a), float(b)
n = int(input('Ge antalet punkter '))
I = mittensumma(f,a,b,n)         # anrop med funktionsnamn
print('Arean är: ',I)

#%% Exempel 8.7
import numpy as np 

def tabell(f,x):
    print('x    f(x)')
    print('---------')
    for i in range(len(x)):
        print(round(x[i],2),'  ',round(f(x[i]),2))
  
# Här börjar själva programmet
f = lambda x: np.sqrt(x)  
x = [1,2,3,4,5]
tabell(f,x)

#%%                                       
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX         XXXXX                              #
#     XX XX       XX XX      XX  XX       XX   XX                             #
#     XXX        XX   XX     XXXXX         XXXXX                              #
#     XX xX     XX XX  XX    XX              XX                               #
#     XX  XX   XX       XX   XX             XX                                #
#                                                                             #
###############################################################################
 
#%% Exempel 9.3   import numpy as np

import numpy as np

print('Uppgift a')
# med for-loop
s = 0                           # initialisera summan till 0
for i in range(1,26):           # loopa äver termer
    s = s + i                   # addera term till summan
print('summa med for-loop:',s)   
# med np.sum
x = np.arange(1,26)             # vektor med heltal från 1 till 25
print('summa med np.sum  :',np.sum(x)) 

print('Uppgift b')
# med foor-loop
s = 0
for i in range(1,101):
    s = s + 1/i
print('summa med for-loop:',s)   
# med np.sum
x = np.arange(1,101)            # vektor med heltal från 1 till 100
print('summa med np.sum  :',np.sum(1/x)) 

print('Uppgift c')
# med for-loop
s = 0
for i in range(5,51):
    s = s + i**2
print('summa med for-loop:',s)
# med np.sum
x = np.arange(5,51)             # vektor med heltal från 5 till 50
print('summa med np.sum  :',np.sum(x**2))  

#%% Exempel 9.4.
import numpy as np
import matplotlib.pyplot as plt

data = np.array([6,3,6,2,5,3,5,6,6,10,4,7,4,2,3,1,12,7,2,3,3,4,10,8,2])
print('Medelvärde: ',np.mean(data))

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
f = np.array([1,4,5,3,2,4,2,1,0,2,0,1])

fig, ax = plt.subplots()
ax.stem(x,f)                               # stolpdiagram
ax.set_xlabel('ordlängd',fontsize = 14)
ax.set_ylabel('frekvens',fontsize = 14)
ax.tick_params(labelsize=14)

#%% Exempel 9.5
import numpy as np
import matplotlib.pyplot as plt

klasser = np.arange(41.5,65.6,3) # klassgränser 41.5-44.5, 44.5-47.5 ets 
data = np.array([53,52,54,56,50,49,52,50,57,52,55,60,51,43,53,64,57,58,57,50,
                 56,50,55,50,59,42,55,45,45,52,46,47,46,58,48,50,55,47,53,51])

print('Medelvärde ',np.mean(data))

fig, ax = plt.subplots()
frekvens, klasser, patch  = ax.hist(data,bins=klasser,edgecolor='black')
ax.set_xlabel('vikt i kg',fontsize=14)
ax.set_ylabel('frekvens',fontsize=14)
ax.tick_params(labelsize=14)
   
print('klassgränser',klasser)
print('frekvens    ',frekvens)

#%% Exempel 9.6
import numpy as np
import matplotlib.pyplot as plt

data = np.array([5.4,5.6,5.8,6.8,5.6,5.0,4.8,3.9,5.3,5.3,
                 5.6,6.4,5.8,4.1,5.0,6.5,5.5,5.7,5.1,5.4,
                 6.0,6.1,5.6,6.5,4.9,4.7,6.7,7.0,6.5,4.9])

print('Medelvärde ',round(np.mean(data),1))

fig, ax = plt.subplots()
frekvens, klasser, patch  = ax.hist(data,bins=7,edgecolor='black')
ax.set_xlabel('temperatur',fontsize=14)
ax.set_ylabel('frekvens',fontsize=14)
ax.tick_params(labelsize=14)

print('klassgränser',klasser)
print('frekvens    ',frekvens)

#%% Exempel 9.7
import numpy as np

l = np.array([179,167,183,179,177,178,182,187,176,172])
print('medelvärde       ',np.mean(l))
print('standardavvikelse',round(np.std(l),1))

#%% Exempel 9.8
import numpy as np
import matplotlib.pyplot as plt

T = np.loadtxt('T.txt')
print('medelvärde       ',round(np.mean(T),1))
print('standardavvikelse',round(np.std(T),1))

fig, ax = plt.subplots()
ax.hist(T,bins=20,edgecolor='black')
ax.set_xlabel('temperatur',fontsize=14)
ax.set_ylabel('frekvens',fontsize=14)
ax.tick_params(labelsize=14)

#%% Exempel 9.9
import numpy as np

l = np.array([31000,33500,35300,34800,255000,39100,32800,30500,34300,38700])
print('medelvärde',np.mean(l))
print('median    ',np.median(l))

#%% Exempel 9.11b
import numpy as np
import matplotlib.pyplot as plt

x = -1 + 4*np.random.randn(100000) # addition av -1 ger medelvärde -1
                                   # multiplikation med 4 ger standaravv. 4
fig, ax = plt.subplots()
ax.hist(x,bins=50,edgecolor='black')
ax.set_xlabel('värden',fontsize=14)
ax.set_ylabel('frekvens',fontsize=14)
ax.tick_params(labelsize=14)

#%% Tillämpning: känslighetsanalys
import numpy as np
import matplotlib.pyplot as plt

N = 100000            # antal slumpvärden
M1  = 6e24            # massa planet 
sM1 = 0.05e24         # standardavvikelse planet 
M2  = 2e30            # massa sol
sM2 = 0.03e30         # standardavvikelse sol
r   = 1.5e11          # avstånd
sr  = 0.05e11         # standardavvikelse avstånd
G   = 6.67384E-11     # gravitationskonstanten

M1_vekt = M1 + sM1*np.random.randn(N)  # vektor med N normalfördelade slumptal
M2_vekt = M2 + sM2*np.random.randn(N)
r_vekt  = r  + sr*np.random.randn(N)
F = G*M1_vekt*M2_vekt/r_vekt**2        

fig,ax = plt.subplots()
ax.hist(F,50,edgecolor='black')
ax.set_xlabel('F',fontsize=14)
ax.set_ylabel('frekvens',fontsize=14)
ax.tick_params(labelsize=14)
print('medelvärde        ',np.mean(F))
print('standardavvikelse ',np.std(F))

#%% Tillämpning: histogram från frekvenstabeller
import matplotlib.pyplot as plt
import numpy as np

klassgrans = np.arange(-0.5,104.6,5)
klassmitt = np.arange(2,102.1,5)
frekvens = np.array([271312,298653,306171,296877,280864,310619,380273, 
                     347937,319679,316405,326231,336975,291833,273918, 
                     263781,263325,177878,106066,52260,15857,2274]) 
medel = np.sum(klassmitt*frekvens)/np.sum(frekvens)
print('Medelålder ',round(medel,1))

fig, ax = plt.subplots()
ax.stairs(frekvens, klassgrans, fill=True)
ax.set_xlabel('ålder',fontsize=14)
ax.set_ylabel('frekvens',fontsize=14)
ax.tick_params(labelsize=14)

#%%
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX       XXX     XXXX                         #
#     XX XX       XX XX      XX  XX       XX    XX  XX                        #
#     XXX        XX   XX     XXXXX        XX   XX    XX                       #
#     XX xX     XX XX  XX    XX           XX    XX  XX                        #
#     XX  XX   XX       XX   XX           XX     XXXX                         #
#                                                                             #
###############################################################################

#%% Exempel 10.2
import matplotlib.pyplot as plt
x = np.linspace(-2,6)
y = 5 - x
fig, ax = plt.subplots()
ax.plot(x,y)
ax.tick_params(labelsize=14)
ax.grid('on')
ax.set_xticks([-2,-1,0,1,2,3,4,5,6])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

#%%
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX       XXX  XXX                             #
#     XX XX       XX XX      XX  XX       XX   XX                             #
#     XXX        XX   XX     XXXXX        XX   XX                             #
#     XX xX     XX XX  XX    XX           XX   XX                             #
#     XX  XX   XX       XX   XX           XX   XX                             #
#                                                                             #
###############################################################################

#%% Exempel 11.7
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.linspace(0,3)
ax.plot(x,x**(-1),label='a = -1')
ax.plot(x,x**0,label='a = 0')
ax.plot(x,x**0.5,label='a = 0.5')
ax.plot(x,x**1,label='a = 1')
ax.plot(x,x**2,label='a = 2')
ax.axis([0,3,0,4])                # begränsar axlarna
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(fontsize=14)

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

#%% Exempel 11.8
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
g = 9.81                   # tyngdaccelerationen
l = np.linspace(0,1,100)   # längd l mellan 0 och 1
T = 2*np.pi*np.sqrt(l/g)   # svängningstid T
ax.plot(l,T)
ax.tick_params(labelsize=14)
ax.grid('on')
ax.set_xlabel('längd i m',fontsize=14)
ax.set_ylabel('tid i s',fontsize=14)

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

#%% Exempel 11.9
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.linspace(-3,3)
ax.plot(x,0.5**x,label='a = 0.5')
ax.plot(x,1**x,label='a = 1')
ax.plot(x,1.5**x,label='a = 1.5')
ax.plot(x,2**x,label='a = 2')
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(fontsize=14)

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

#%% Exempel 11.10
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.linspace(0,50)
K = 1000
ax.plot(t,K*1.05**t)           # exponentiellt växande funktion
ax.tick_params(labelsize=14)
ax.grid('on')
ax.set_xlabel('tid i år',fontsize=14)
ax.set_ylabel('kapital',fontsize=14)

#%% Exempel 11.11
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.linspace(-3,3)
ax.plot(x,np.exp(-x),label='k = -1')
ax.plot(x,np.exp(-0.5*x),label='k = -0.5')
ax.plot(x,np.exp(0.5*x),label='k = 0.5')
ax.plot(x,np.exp(x),label='k = 1')
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(fontsize=14)

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

#%% Exempel 11.12
import numpy as np
import matplotlib.pyplot as plt

# inför täthetsfunktionen som en anonym funktion
f = lambda x, mu, sigma : 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2/(2*sigma**2))

fig, ax = plt.subplots()
x = np.linspace(-5,5,200)
ax.plot(x,f(x,0,0.5),label=r'$\mu = 0, \sigma = 0.5$')
ax.plot(x,f(x,0,1),label=r'$\mu = 0, \sigma = 1$')
ax.plot(x,f(x,0,2),label=r'$\mu = 0, \sigma = 2$')
ax.plot(x,f(x,-2,1),label=r'$\mu = -2, \sigma = 1$')
ax.grid('on')
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)

#%% Tillämpning: befolkningstillväxt
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.linspace(0,50)
ax.plot(t,41.1*1.034**t,label='Af.')  # Afganistan faktor 1.034
ax.plot(t,144.2*0.99**t,label='Ru.')  # Ryssland   faktor 0.99
ax.tick_params(labelsize=14)
ax.grid('on')
ax.set_xlabel('tid i år',fontsize=14)
ax.set_ylabel('befolkning',fontsize=14)
ax.legend(fontsize=14)

#%% Tillämpning: radioaktivt nedfall
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.linspace(0,100)
lambda_Cs = 0.0231
N0 = 100
ax.plot(t,N0*np.exp(-lambda_Cs*t))
ax.tick_params(labelsize=14)
ax.grid('on')
ax.set_xlabel('tid i år',fontsize=14)
ax.set_ylabel('mängd',fontsize=14)

#%%
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX       XXX    XXXX                          #
#     XX XX       XX XX      XX  XX       XX       XX                         #
#     XXX        XX   XX     XXXXX        XX     XX                           #
#     XX xX     XX XX  XX    XX           XX    XX                            #
#     XX  XX   XX       XX   XX           XX   XXXXXX                         #
#                                                                             #
###############################################################################

#%% Exempel 12.1
import numpy as np
import matplotlib.pyplot as plt

# Plottar till vänster
fig, ax = plt.subplots()
t = np.linspace(-3,3,200)
ax.plot(t,3 + 4*np.sin(2*t),label = '3 + 4sin(2t)')
ax.plot(t,3 + 4*np.sin(4*t),label = '3 + 4sin(4t)')
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(loc='upper right',fontsize=14)
# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# plottar till höger
fig, ax = plt.subplots()
t = np.linspace(-3,3,200)
ax.plot(t,3 + 4*np.sin(2*t),  label = '3 + 4sin(2t)')
ax.plot(t,3 + 4*np.sin(2*t-1),label = '3 + 4sin(2t-1)')
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(loc='upper right',fontsize=14)
# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero') 

#%% Exempel 12,2
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = np.linspace(-3,3,200)
ax.plot(t,3*np.sin(2*t),label = '3sin(2t)')
ax.plot(t,4*np.cos(2*t),label = '4cos(2t)')
ax.plot(t,3*np.sin(2*t)+4*np.cos(2*t),label = '3sin(2t) + 4cos(2t)')
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(loc='upper left',fontsize=14)
# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

#%% Tillämpning: medeltemperatur
import numpy as np
import matplotlib.pyplot as plt

tdata = np.arange(1,13)
Mdata = np.array([-0.6,-0.5,1.9,6.0,11.4,15.4,16.8,16.5,13.0,9.1,4.5,1.1])

fig, ax = plt.subplots()
ax.plot(tdata,Mdata,'o')
t = np.linspace(1,12,100)
M0 = np.mean(Mdata)     # medelvärde av data
A = (16.8 - (-0.6))/2   # högsta minus lägsta delat med två
omega = np.pi/6         # en full svängning  på 12 månader
delta = -2.2            # fått fram via testning
M = M0 + A*np.sin(omega*t + delta)
ax.plot(t,M)
ax.tick_params(labelsize=14)
ax.set_xlabel('tid i månader',fontsize=14)
ax.set_ylabel('medeltemperatur M',fontsize=14)
print('M0 = ',round(M0,2))
print('A  = ',round(A,2))

#%%
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX     XXX   XXXX                             #
#     XX XX       XX XX      XX  XX     XX      XX                            #
#     XXX        XX   XX     XXXXX      XX     XX                             #
#     XX xX     XX XX  XX    XX         XX      XX                            #
#     XX  XX   XX       XX   XX         XX   XXXX                             #
#                                                                             #
###############################################################################

#%% Exempel 13.1
import numpy as np  
import matplotlib.pyplot as plt
import scipy.optimize as opt

fig, ax = plt.subplots()	
t = np.linspace(0,5,100)
f = lambda t : 10*np.exp(-3*t) + 2*np.exp(-5*t)
ax.plot(t,f(t))
ax.grid()
ax.tick_params(labelsize=14)

g = lambda t : f(t) - 6   # ny funk. g(t) där vi har subtraherat 6 från f(t) 
r = opt.root(g,0.2)       # lös ekvationen g(t) = 0 med startvärde t = 0.2
                          # spara information om lösandet i variabeln r 
print(r.message)          # r.message informerar om rot har hittats
print(r.x)                # r.x anger värdet på roten

#%% Exempel 13.2
import numpy as np
import matplotlib.pyplot as plt

def roots(x,f):
    """ 
    x vektor med x-värden i intervallet [a,b]
    f vektor med motsvarande funktionsvärden
    r är en lista med rötter till ekvationsn f(x) = 0
    """
    r = []    # tom lista
    for i in range(len(x)-1):
        if f[i]*f[i+1] <= 0:                # f[i], f[i+1] olika tecken om 
            print('Rot ',(x[i+1] + x[i])/2) # produkten negativ
            r.append((x[i+1] + x[i])/2)     # om rot, addera till listan
    return r

x = np.linspace(-2,2,1000)                  # 1000 punkter i intervall [-2,2]
f = 2*x**3 - 8*x                            # f vektor med funktionsvärden
fig, ax = plt.subplots()
ax.plot(x,f)
ax.plot(x,2 + 0*x)                          # plotta y = 2
ax.grid('on')
ax.tick_params(labelsize=14)

# kalla på roots med vektorn f - 2
r = roots(x,f - 2)

#%% Tillämpning: jordens befolkning exponentiell modell
import numpy as np  
import matplotlib.pyplot as plt
import scipy.optimize as opt

fig, ax = plt.subplots()	
t = np.linspace(0,50,100)
f = lambda t : 8.099*1.009**t
ax.plot(t,f(t))
ax.grid()
ax.tick_params(labelsize=14)

g = lambda t : f(t) - 10   # ny funk. g(t) där vi har subtraherat 10 från f(t)  
r = opt.root(g,23)         # lös ekvationen g(t) = 0 med startvärde t = 23
                           # r ger en massa information om lösandet
print(r.message)           # r.message informerar om rot har hittats
print(r.x)                 # r.x anger värdet på roten

#%%                    
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX     XXX   XX  XX                           #
#     XX XX       XX XX      XX  XX     XX   XX  XX                           #
#     XXX        XX   XX     XXXXX      XX   XXXXXX                           #
#     XX xX     XX XX  XX    XX         XX       XX                           #
#     XX  XX   XX       XX   XX         XX       XX                           #
#                                                                             #
###############################################################################

#%% Exempel 14.1a
import numpy.polynomial.polynomial as pol
import numpy as np
xdata = np.array([0,1,2,3,4]) 
ydata = np.array([0.01,0.91,2.02,3.12,4.15])
p1 = pol.polyfit(xdata,ydata,1)
print(p1)

import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(xdata,ydata,'+',ms=15) # plotta data
x = np.linspace(-1,5)          # tät gridd för modellfunk.
y = p1[0] + p1[1]*x
ax.plot(x,y)                   # plotta modellfunktionen
ax.set_xlabel('x',fontsize=14)
ax.set_ylabel('y',fontsize=14)
ax.tick_params(labelsize=14)  

#%% Exempel 14.1b
p3 = pol.polyfit(xdata,ydata,3)
print(p3)
fig, ax = plt.subplots()
ax.plot(xdata,ydata,'+',ms=15) # plotta data
x = np.linspace(-1,5)          # tät gridd för modellfunk.
y = p3[0] + p3[1]*x + p3[2]*x**2 + p3[3]*x**3
ax.plot(x,y)                   # plotta modellfunktionen
ax.set_xlabel('x',fontsize=14)
ax.set_ylabel('y',fontsize=14)
ax.tick_params(labelsize=14)  

#%% Exempel 14.2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
	
def f(A,t):                      # modellfunktion
    return A[0]*np.exp(-A[1]*t)		
	
def res(A,tdata,Adata):          # residual
    return Adata - f(A,tdata)
			
tdata = np.array([0,1.25,2.5,3.75,5,6.25,7.5,8.75,10,\
                  11.25,12.5,13.75,15.])
Adata = np.array([513.0,349,196,124,83,72,43,38,23,22,\
                  12,5,10])
A0 = [500,0.3]                   # startgissning
A,q = opt.leastsq(res,A0,(tdata,Adata))
print('Parametrar:',A)
fig, ax = plt.subplots()
ax.plot(tdata,Adata,'+',ms=15)   # plotta data
t = np.linspace(0,15,200)        # tät gridd för modellfunk.
A = f(A,t)
ax.plot(t,A)                     # plotta modellfunktionen
ax.set_xlabel('t',fontsize=14)
ax.set_ylabel('A',fontsize=14)
ax.tick_params(labelsize=14)  

#%% Exempel 14.3
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

xdata = np.array([0,40,80,120,160,200,250,300,400,500,600,700])
hdata = np.array([0,36,65,85,100,113,135,141,160,165,175,185])	
	
def f(a,x):                      # modellfunktion
    return a[0]*x/(1 + a[1]*x)	
def res(a,xdata,hdata):          # residual
    return hdata - f(a,xdata)

a0 = [0.8,0.004]                 # startgissning
a,q = opt.leastsq(res,a0,(xdata,hdata))
print('Parametrar:',a)
fig, ax = plt.subplots()
ax.plot(xdata,hdata,'+',ms=15)   # plotta data
x = np.linspace(0,1000,200)      # tät gridd för modellfunk.
h = f(a,x)
ax.plot(x,h)                     # plotta modellfunktionen
ax.set_xlabel('x i cm',fontsize=14)
ax.set_ylabel('h i cm',fontsize=14)
ax.tick_params(labelsize=14)  

#%% Tillämpning: jordens befolkning logistisk modell
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def f(a,t):                          # modellfunktion
    num   = a[0]*a[1]*np.exp(a[2]*t)
    denom = a[1]*np.exp(a[2]*t) + (a[0] - a[1])
    return num/denom

def res(a,tdata,Ndata):              # residual
    return Ndata - f(a,tdata)

Ndata = np.loadtxt("population.txt") # läs data från fil
tdata = np.arange(73)                # t = 0, år 1951, t = 72, år 2023 

fig, ax = plt.subplots()             # plotta befolkningsdata 1951 till 2023
ax.plot(tdata,Ndata,'+')

a0 = [12,2.54,0.02]                  # startgissning
a,q = opt.leastsq(res,a0,(tdata,Ndata))
print('Parametrar:',a)

t = np.linspace(0,149)               # tid fram till 2100, dvs t = 149
N = f(a,t)
ax.plot(t,N)                         # plotta modellfunktionen
ax.set_xlabel('t i år från 1951',fontsize=14)
ax.set_ylabel('befolkning miljarder',fontsize=14)
ax.tick_params(labelsize=14)   

#%% Tillämpning: CO$_2$ i atmosfären
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

k = 2*np.pi
def f(a,t):                      # modellfunktion
    return a[0] + a[1]*t + a[2]*t**2 + \
           a[3]*np.sin(k*t)   + a[4]*np.cos(k*t) + \
           a[5]*np.sin(2*k*t) + a[6]*np.cos(2*k*t)

def res(a,tdata,co2data):        # residual
    return co2data - f(a,tdata)

co2 = np.loadtxt('co2full2010.txt',usecols=(2,3))
tdata = co2[:,0]                 # kolonn 1 är tiden
co2data = co2[:,1]               # kolonn 2 är co2

a0 = [390,2.5,0,0,0,0,0]         # startgissning
a,q = opt.leastsq(res,a0,(tdata,co2data))
print('Parametrar:',a)
fig, ax = plt.subplots()
ax.plot(tdata,co2data,'+')       # plotta modellfunktionen
t = np.linspace(2010,2022,1000)
y = f(a,t)
ax.plot(t,y)   
ax.set_xlabel('t',fontsize=14)
ax.set_ylabel('co2 in ppm',fontsize=14)
ax.tick_params(labelsize=14)

t = 2050
co2_2050 = a[0] + a[1]*t + a[2]*t**2
print('CO2 år 2050: ',round(co2_2050,1),'ppm')

#%%
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX     XXX    XXXXX                           #
#     XX XX       XX XX      XX  XX     XX   XX                               #
#     XXX        XX   XX     XXXXX      XX   XXXXXX                           #
#     XX xX     XX XX  XX    XX         XX       XX                           #
#     XX  XX   XX       XX   XX         XX   XXXXX                            #
#                                                                             #
###############################################################################

#%% Exempel 15.5

import numpy as np
import matplotlib.pyplot as plt

# funktion för att beräkna derivatan med hjälp av finita differenser
def fdiff(f,h):
    """
    beräkna derivatan f'(x) m.h.a finita differenser
    f vektor med funktionsvärden på en gridd
    h griddavstånd
    df vektor med derivator på en gridd
    """
    n = len(f)                               # antalet griddpunkter
    df = np.zeros(n)                         # vektor med derivatan
    for i in range(n):                       # loopa över griddpunkterna
        if i == 0:                           # första punkten
            df[i] = (f[i+1] - f[i])/h        # framåtdifferens
        elif i == n - 1:                     # sista punkten
            df[i] = (f[i] - f[i-1])/h        # bakåtdifferens
        else:                                # inre punkter
            df[i] = (f[i+1] - f[i-1])/(2*h)  # centraldifferens
    return df
 
x = np.linspace(0,5,2000)                    # tät gridd
h = x[1] - x[0]                              # griddavstånd
f = x**2/(1+x**2)                            # vektor med funktionsvärden

fig, ax = plt.subplots()
ax.plot(x,f,label="f(x)")                    # plotta funktionen
ax.plot(x,fdiff(f,h),label="f'(x)")          # fdiff(f,h) vektor med dervatan
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(fontsize=14)

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# bestäm maxvärde för derivatan och i vilken punkt
print("Maxvärde för f'(x) ",round(np.max(fdiff(f,h)),2))
print("Maxvärde för f'(x) antas i x = ",\
      round(x[np.argmax(fdiff(f,h))],2))

import numpy as np
import matplotlib.pyplot as plt

n = 2000
x = np.linspace(0,5,n)
h = x[1] - x[0]                         # griddavstånd
f = x**2/(1+x**2)          
df = np.zeros(n)                        # inför derivatan df som en nollvektor
for i in range(n):                      # loopa över griddpunkterna
    if i == 0:
        df[i] = (f[i+1] - f[i])/h       # framåtdifferens
    elif i == n - 1:
        df[i] = (f[i] - f[i-1])/h       # bakåtdifferens
    else:
        df[i] = (f[i+1] - f[i-1])/(2*h) # centraldifferens

fig, ax = plt.subplots()
ax.plot(x,f,label="f(x)")               # plotta funktionen
ax.plot(x,df,label="f'(x)")             # plotta derivatan
ax.tick_params(labelsize=14)
ax.grid('on')
ax.legend(fontsize=14)

# dessa kommandon ger axlar genom origo
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# bestäm maxvärde för derivatan och i vilken punkt
print("Maxvärde för f'(x) ",round(np.max(df),2))
print("Maxvärde för f'(x) antas i x = ",round(x[np.argmax(df)],2))

#%% Tillämpning: befolkningstillväxt
import numpy as np
import matplotlib.pyplot as plt

# funktion för att beräkna derivatan med hjälp av finita differenser
def fdiff(f,h):    
    """
    beräknar derivatan f'(x) m.h.a finita differenser
    f vektor med funktionsvärden på en gridd
    h griddavstånd
    df vektor med derivator på en gridd
    """
    n = len(f)                               # antalet griddpunkter
    df = np.zeros(n)                         # vektor med derivatan
    for i in range(n):                       # loopa över griddpunkterna
        if i == 0:                           # första punkten
            df[i] = (f[i+1] - f[i])/h        # framåtdifferens
        elif i == n - 1:                     # sista punkten
            df[i] = (f[i] - f[i-1])/h        # bakåtdifferens
        else:                                # inre punkter
            df[i] = (f[i+1] - f[i-1])/(2*h)  # centraldifferens
    return df
 
t = np.linspace(0,149,2000)                  # tät gridd
h = t[1] - t[0]                              # griddasvtånd
a = 12.459*2.494*np.exp(0.0277*t)            # täljare
b = 2.494*np.exp(0.0277*t) + (12.459-2.494)  # nämnare
N = a/b                                      # vektor med funktionsvärde
 
fig, ax = plt.subplots()
ax.plot(t,fdiff(N,h),label="N'(t)")          # fdiff(N,h) vektor med dervatan
ax.tick_params(labelsize=14)
ax.set_xlabel('t i år från 1951',fontsize=14)
ax.set_ylabel('befolkningsförändring per år i miljarder',fontsize=14)
ax.grid('on')
ax.legend(fontsize=14)

# bestäm maxvärde för derivatan och i vilken punkt
dmax = np.max(fdiff(N,h))          # maximalt värde på derivatan
imax = np.argmax(fdiff(N,h))       # index för maximalt värde
tmax = t[imax]                     # motsvarande tid

print("Maximal befolkningsförändring per år ",round(dmax,5),"miljarder")
print("Maximal befolkningsförändring för t = ",round(tmax))

# bestäm maximal årlig tillväxt i procent (relativ förändring) 
print("Maximal årligt tillväxt ",round(100*dmax/N[imax],3),"procent")

#%%                                       
###############################################################################
#                                                                             #
#     XX  XX       XX        XXXXX     XXX    XXXXXX                          #
#     XX XX       XX XX      XX  XX     XX   XX                               #
#     XXX        XX   XX     XXXXX      XX   XX XXXX                          #
#     XX xX     XX XX  XX    XX         XX   XX    XX                         #
#     XX  XX   XX       XX   XX         XX    XXXXXX                          #
#                                                                             #
###############################################################################

#%% Exempel 16.4a
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr

f = lambda t, N: -N                  # obs, t måste med
sol = intgr.solve_ivp(f,[0,3],[1])  # [1] och inte bara 1.
print(sol)

#%%

f = lambda t, N: -N           # obs, t måste med
sol = intgr.solve_ivp(f,[0,3],[1],t_eval=[0,1,2,3])
print(sol)

#%%

f = lambda t, N: -N
sol = intgr.solve_ivp(f,[0,3],[1],\
      t_eval=np.linspace(0,3,100))     # tät gridd för plottning
fig, ax = plt.subplots()
ax.plot(sol.t,sol.y[0])                # sol.y är en matris, lösningen 
                                       # ges i första raden
ax.set_xlabel('t',fontsize=14)
ax.set_ylabel('N(t)',fontsize=14)
ax.tick_params(labelsize=14)

#%% Exempel 16.4b
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr

f = lambda t, T: -0.02*(T - 20)
sol = intgr.solve_ivp(f,[0,150],[100],\
      t_eval=np.linspace(0,150,100))     # tät gridd för plottning
fig, ax = plt.subplots()
ax.plot(sol.t,sol.y[0])                  # sol.y är en matris, lösningen 
                                         # ges i första raden
ax.set_xlabel('t minuter',fontsize=14)
ax.set_ylabel('T(t)',fontsize=14)
ax.tick_params(labelsize=14)

#%% Exempel 16.5
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr
import scipy.optimize as opt

# modellfunktion fås som lösning till diffekv
# lösning räknas ut för de t vi har i tabellen (tdata)
def fmodell(a,t):                      
    f = lambda t, T: -a[0]*(T - a[1])   # högerled i diffekv
    sol = intgr.solve_ivp(f,[0,126],[a[2]],t_eval = tdata)  
    return sol.y[0]

def res(a,tdata,Tdata): # residual
    return Tdata - fmodell(a,tdata)

tdata = np.array([0,10,20,40,63,90,126])
Tdata = np.array([68,56.5,49,40,33.5,29,26])

a0 = [0.02,20,68]       # startgissning

a,q = opt.leastsq(res,a0,(tdata,Tdata))
print('Parametrar:',a)

# plotta modellfunktion tillsammans med uppmätta data 
fig, ax = plt.subplots()
ax.plot(tdata,Tdata,'o')
t = np.linspace(0,150,200)
f = lambda t, T: -a[0]*(T - a[1])
sol = intgr.solve_ivp(f,[0,150],[a[2]],t_eval = t)  # tät gridd för plottning
ax.plot(t,sol.y[0])
ax.set_xlabel('t minuter',fontsize=14)
ax.set_ylabel('T(t)',fontsize=14)
ax.tick_params(labelsize=14)

#%% avsnitt 16.5.1
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr

fig, ax = plt.subplots()
r = 0.03
f = lambda t, N: r*N
sol = intgr.solve_ivp(f,[0,50],[100000],\
      t_eval=np.linspace(0,50,100))       # tät gridd för plottning
ax.plot(sol.t,sol.y[0],label='r > 0')     # sol.y är en matris, lösningen 
                                          # ges i första raden
r = -0.03
f = lambda t, N: r*N
sol = intgr.solve_ivp(f,[0,50],[100000],\
      t_eval=np.linspace(0,50,100))       # tät gridd för plottning
ax.plot(sol.t,sol.y[0],label='r < 0')     # sol.y är en matris, lösningen 
                                          # ges i första raden
ax.set_xlabel('t',fontsize=14)
ax.set_ylabel('N(t)',fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)

#%% avsnitt 16.5.2
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr

fig, ax = plt.subplots()
K = 1000
N0 = 10
# råtta
rmax = 0.6*0.250**(-0.25)
f = lambda t, N: rmax*(1 - N/K)*N
sol = intgr.solve_ivp(f,[0,20],[N0],\
      t_eval=np.linspace(0,20,100))        # tät gridd för plottning
ax.plot(sol.t,sol.y[0],label='råtta')      # sol.y är en matris, lösningen 
                                           # ges i första raden
# vildsvin
rmax = 0.6*100**(-0.25)
f = lambda t, N: rmax*(1 - N/K)*N
sol = intgr.solve_ivp(f,[0,20],[N0],\
      t_eval=np.linspace(0,20,100))        # tät gridd för plottning
ax.plot(sol.t,sol.y[0],label='vildsvin')   # sol.y är en matris, lösningen 
                                           # ges i första raden
ax.set_xlabel('t',fontsize=14)
ax.set_ylabel('N(t)',fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)

#%% avsnitt 16.5.3

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr
import scipy.optimize as opt

# modellfunktion fås som lösning till diffekv
# lösning räknas ut för de t vi har i tabellen (tdata)
def fmodell(a,t): 
    f = lambda t, N: a[0]*(1 - N/a[1])*N # högerled i diffekv
    sol = intgr.solve_ivp(f,[0,19],[a[2]],t_eval = tdata) 
    return sol.y[0]

def res(a,tdata,Ndata): # residual
    return Ndata - fmodell(a,tdata)

tdata = np.arange(0,19)
Ndata = np.array([2,10,17,29,39,63,185,258,267,392,\
                  510,570,650,560,575,550,480,520,500])

a0 = [1,500,2] # startgissning
a,q = opt.leastsq(res,a0,(tdata,Ndata))
print('Parametrar:',a)
# plotta modellfunktion tillsammans med uppmätta data
fig, ax = plt.subplots()
ax.plot(tdata,Ndata,'o')

t = np.linspace(0,19,200)

f = lambda t, N: a[0]*(1 - N/a[1])*N

sol = intgr.solve_ivp(f,[0,19],[a[2]],t_eval = t) # tät gridd för plottning
ax.plot(t,sol.y[0])
ax.set_xlabel('t minuter',fontsize=14)
ax.set_ylabel('antal per cm-3',fontsize=14)
ax.tick_params(labelsize=14)

#%% Exempel 16.8
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr

lam0 = 0.138 
lam1 = 0.00501
f = lambda t,N : [-lam0*N[0],                
                   lam0*N[0] - lam1*N[1]] 

sol = intgr.solve_ivp(f,[0,200],[10000,0],t_eval=np.linspace(0,200,100))
fig, ax = plt.subplots()
ax.plot(sol.t,sol.y[0],label=r"$^{210}$Bi")  # N0 första raden sol.y[0]
ax.plot(sol.t,sol.y[1],label=r"$^{210}$Po")  # N1 andra raden sol.y[1]
ax.set_xlabel("t i dagar",fontsize=14)
ax.set_ylabel("N",fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)

#%% Exempel 16.9
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as intgr

fig, ax = plt.subplots()

# I = 0
f = lambda t,M : [-92/750*M[0] + 90/38000*M[1],
                   92/750*M[0] - 90/38000*M[1]]
sol = intgr.solve_ivp(f,[0,50],[750,38000],t_eval=np.linspace(0,50,100))
ax.plot(sol.t,sol.y[0],label='I = 0')  # plotta mängd i atmosfären

# I = 6
f = lambda t,M : [-92/750*M[0] + 90/38000*M[1] + 6,
                   92/750*M[0] - 90/38000*M[1]]
sol = intgr.solve_ivp(f,[0,50],[750,38000],t_eval=np.linspace(0,50,100))
ax.plot(sol.t,sol.y[0],label='I = 6')  # plotta mängd i atmosfären

ax.set_xlabel("t i år",fontsize=14)
ax.set_ylabel("mängd i miljarder ton",fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14)

#%% tillämpning: smittspridning
import numpy as np
import scipy.integrate as intgr
import matplotlib.pyplot as plt

# Populationsstorlek
N = 1000
# Initialt antal smittade, I0, och återhämtade, R0, personer
I0 = 1
R0 = 0
# Alla övriga är mottagliga så initialt har vi
S0 = N - I0 - R0

beta = float(input('Ge kontakthastighet '))
recovertime = float(input('Återhämtningstid i dagar '))
gamma = 1/recovertime 


# Modellens differentialekvationer
f = lambda t, y: [-beta*y[0]*y[1]/N,
                   beta*y[0]*y[1]/N - gamma*y[1], 
                   gamma*y[1]]

# Tidsgridd i dagar
t = np.linspace(0, 160, 160)
# Integrera ekvationerna på tidsgridden.
sol = intgr.solve_ivp(f,[0,160],[S0,I0,R0],t_eval = t)

# Plotta S(t), I(t) och R(t)
fig, ax = plt.subplots() 
ax.plot(sol.t,sol.y[0]/1000,label='Susceptible')
ax.plot(sol.t,sol.y[1]/1000,'--',label='Infected')
ax.plot(sol.t,sol.y[2]/1000,':',label='Recovered with immunity')
ax.set_xlabel('Time /days',fontsize=14)
ax.set_ylabel('Numbers (1000s)',fontsize=14)
ax.set_ylim(0,1.2)
ax.legend(fontsize=14)
ax.tick_params(labelsize=14)

#%%
import numpy as np
import scipy.integrate as intgr
import matplotlib.pyplot as plt

# Populationsstorlek
N = 1000

beta = float(input('Ge kontakthastighet '))
recovertime = float(input('Återhämtningstid i dagar  '))
gamma = 1/recovertime 
percent = float(input('Procent av befolkningen som är vaccinerade '))
v = float(input('Vaccinationshastighet '))

# Initialt antal smittade, I0, återhämtade, R0,
# och vaccinated, V0, personer
I0 = 1
R0 = 0
V0 = percent*0.01*N   
# Alla övriga är mottagliga så initialt har vi
S0 = N - I0 - R0 - V0

# Modellens differentialekvationer
f = lambda t, y: [-beta*y[0]*y[1]/N - v*y[0],
                   beta*y[0]*y[1]/N - gamma*y[1], 
                   gamma*y[1],
                   v*y[0]]

# Tidsgridd i dagar
t = np.linspace(0, 160, 160)
# Integrera ekvationerna på tidsgridden.
sol = intgr.solve_ivp(f,[0,160],[S0,I0,R0,V0],t_eval = t)

# Plotta S(t), I(t), R(t) och V(t)
fig, ax = plt.subplots() 
ax.plot(sol.t, sol.y[0]/1000,label='Susceptible')
ax.plot(sol.t, sol.y[1]/1000,'--',label='Infected')
ax.plot(sol.t, sol.y[2]/1000,':',label='Recovered with immunity')
ax.plot(sol.t, sol.y[3]/1000,'-.',label='Vaccinated')
ax.set_xlabel('Time /days',fontsize=14)
ax.set_ylabel('Numbers (1000s)',fontsize=14)
ax.set_ylim(0,1.2)
ax.legend(fontsize=14)
ax.tick_params(labelsize=14)
