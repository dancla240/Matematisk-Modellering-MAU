# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:05:46 2024

@author: danie
"""
#%% Importerar librarys
import numpy as np
import matplotlib.pyplot as plt
#%% Uppgift 1
mens_length = np.loadtxt('men_length.txt')
women_length = np.loadtxt('women_length.txt')

print('Men max:', np.max(mens_length))
print('Men min:', np.min(mens_length))
print('Men mean:', round(np.mean(mens_length), 1))
print('Men stddev:', round(np.std(mens_length), 1))
# %% Uppgift 1 plottar
fig, ax = plt.subplots()
ax.hist(mens_length, bins=20)
ax.set_ylabel('lenght')
ax.set_xlabel('frequence')
ax.set_title('Histogram Mens Lenght')

# %%
