# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:55:12 2020

@author: Mikko Impiö
"""

import matplotlib.pyplot as plt
import numpy as np

fontsize=14

y = [50, 40, 20, 10, 8, 5, 4, 2, 0.5, 0.5, 0.5, 0.5]
x = np.arange(len(y))

plt.plot(x,y)
plt.axis([0,len(y)-1,0,55])

plt.xticks(np.arange(0,len(y)), np.arange(0,len(y))+1)
numbers = ["I", "III", "V", "VII", "IX", "XI", "XII", "XV"]

for label in plt.gca().xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
    
    
labels = [item.get_text() for item in plt.gca().get_xticklabels()]
ii = 0
for i, label in enumerate(labels):
    if i%2==0:
        labels[i] = numbers[ii]
        ii = ii+1
        
plt.gca().set_xticklabels(labels, fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.fill_between([4,5], 0, y[4:6], color='g', alpha=0.6)

plt.xlabel("Abundance class", fontsize=fontsize)
plt.ylabel("Percent of species in abundance class",fontsize=fontsize)

plt.annotate("Most species are in the \nrare species abundance classes",
             xy=(1, 44),
             fontsize=fontsize-2)

plt.annotate("Indicator species",
             xy=(4.5,6.5),
             xytext=(3,25),
             fontsize=fontsize,
             arrowprops=dict(arrowstyle="->"))

plt.annotate("Most populous classes \ncontain only \na few species",
             xy=(6.8, 5),
             fontsize=fontsize-2)