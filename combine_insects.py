# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:45:13 2020

@author: Mikko Impiö
"""

import re
import os
import numpy as np

def folder_from_fpath(fpath):
    folder = os.path.split(os.path.split(fpath)[0])[1]
    return folder
    
def num_from_insect(x):
    try:
        ret = int(re.findall(r'\d+',x)[0])
    except:
        ret = 0 
    return  ret
    
def add_insect_class(df):
    ddf = df.copy()

    ddf['insect'] = ddf.iloc[:,0].apply(lambda x: num_from_insect(folder_from_fpath(x))) #find the number of the insect 
    ddf['insectname'] = ddf.iloc[:,0].apply(lambda x: folder_from_fpath(x))
    return ddf

def add_yhat(df, yhat):
    ddf = df.copy()
    ddf['pred'] = yhat
    return ddf


def get_gt(ds, length):
    labels = []
    i = 1
    for _, label in ds:
        labels.append(int(np.argmax(label.numpy())+1))
        print("{}/{}".format(i,length))
        i += 1
    return np.asarray(labels)
