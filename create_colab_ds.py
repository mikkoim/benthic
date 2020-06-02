# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:17:30 2020

@author: Mikko Impiö
"""

import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import tensorflow as tf
from shutil import copyfile

from loadbm import create_df, create_tf_dataset, prepare_for_training

#Load the ready-made splits


if platform.system() == 'Linux':
    datapath = '/home/mikko/Documents/kandi/data/IDA/Separate lists with numbering/Machine learning splits'
    img_path = '/home/mikko/Documents/kandi/data/IDA/Images/'
else:
    datapath = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Separate lists with numbering\\Machine learning splits'
    img_path = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Images\\'

split = 1

train_fname = 'train'+str(split)+'.txt'
test_fname = 'test'+str(split)+'.txt'
val_fname = 'val'+str(split)+'.txt'

part_dat = True

df_train = create_df(os.path.join(datapath, train_fname),
                     img_path,
                     partial_dataset=part_dat,
                     seed=123)

df_test = create_df(os.path.join(datapath, test_fname),
                     img_path,
                     partial_dataset=part_dat,
                     seed=123)

df_val = create_df(os.path.join(datapath, val_fname),
                     img_path,
                     partial_dataset=part_dat,
                     seed=123)

#%%
from PIL import Image
root = 'colab_ds'

def create_colab_ds(df):
    for i, fpath in enumerate(df.path):
        dirpath, filename = os.path.split(fpath)
        d = os.path.split(dirpath)[1]
        
        target_dir = os.path.join(root, d)
        
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        
        target_fname = os.path.join(target_dir, filename)
        
        img = Image.open(fpath)
        img.save(target_fname)
        
        print('{}/{}'.format(i+1, len(df.path)))    
    

create_colab_ds(df_train)
create_colab_ds(df_test)
create_colab_ds(df_val)

df_train.to_pickle('df_train.pkl')
df_test.to_pickle('df_test.pkl')
df_val.to_pickle('df_val.pkl')