#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:28:44 2020

@author: mikko
"""

import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


## TXT FILE TO DATAFRAME

def open_log(fname):
    with open(fname) as file:
        
        lsraw = file.readlines()
        
    m = '[==============================]'
    

    di = defaultdict(list)
    for l in lsraw:
        if l.find(m) != -1:
            parts = l.split(' - ')
            parts = [x.replace('\n','') for x in parts[2:]]
            
            metrics = [(x.split(' ')[0][0:-1], x.split(' ')[1]) for x in parts]
            
            for k, v in metrics:
                di[k].append(v)
        
    d = dict(di)
    
    df = pd.DataFrame(d).astype(np.float32)
    return df
    
    
logfolder = os.path.abspath('logs')

df_ref = open_log(os.path.join(logfolder, '26-01-2020_colab.txt'))
df = open_log(os.path.join(logfolder, '31-xx-2020.txt'))

#%%

fname = '18-01-2020_cont_colab.log'

df_ref = pd.read_csv(os.path.join(logfolder, fname))

#%%

plt.subplot(2,1,1)
plt.plot(df['loss'], linestyle='dashed', color='r')
plt.plot(df['val_loss'], color='r')
plt.legend(['loss', 'val loss'])

plt.subplot(2,1,2)
plt.plot(df['accuracy'], linestyle='dashed', color='r')
plt.plot(df['val_accuracy'], color='r')
plt.legend(['acc', 'val acc'])

#%%

plt.subplot(2,1,1)
plt.plot(df_ref['loss'], linestyle='dashed', color='b')
plt.plot(df_ref['val_loss'], color='b')

plt.plot(df['loss'], linestyle='dashed', color='r')
plt.plot(df['val_loss'], color='r')

plt.legend(['loss ref', 'val loss ref', 'loss', 'val loss'])

plt.subplot(2,1,2)
plt.plot(df_ref['accuracy'], linestyle='dashed', color='b')
plt.plot(df_ref['val_accuracy'], color='b')

plt.plot(df['accuracy'], linestyle='dashed', color='r')
plt.plot(df['val_accuracy'], color='r')

plt.legend(['acc ref', 'val acc ref', 'acc', 'val acc'])



