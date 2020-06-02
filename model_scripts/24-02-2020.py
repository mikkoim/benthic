# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:05:04 2020

@author: Mikko Impiö
"""

import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

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

part_dat = False

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

from sklearn.utils import shuffle

df_train = shuffle(df_train)
df_val = shuffle(df_val)

h = np.histogram(df_train['label'],bins=39)[0]
plt.bar(range(1,40), h)
plt.title('pre-sampling')

#%% Data augmentation by oversampling

def calc_sampling(df, method='mean', resample='over'):
    h = np.histogram(df['label'],bins=39)[0]
    if method=='mean':
      a = np.mean(h).astype(int)
    elif method=='max':
      a = np.max(h).astype(int)
    else: 
        raise Exception

    delta = a-h
    
    if resample == 'over':
        delta[delta <0] = 0
    elif resample == 'under':
        delta[delta >0] = 0
        delta = np.abs(delta)
    else:
        raise Exception

    return delta

def oversample_df(df, delta):
    lst = [df]
    for i, group in df.groupby('label'):
        lst.append(group.sample(delta[i-1], replace=True))
        
    return pd.concat(lst)

def undersample_df(df):
    
    h = np.histogram(df['label'],bins=39)[0]
    a = np.mean(h).astype(int)
    
    lst = []
    for i, group in df.groupby('label'):
        
        if a>len(group):
            lst.append(group)
        else:
            lst.append(group.sample(a, replace=False))
        
    return pd.concat(lst)

df_train_new = oversample_df(df_train, 
                         calc_sampling(df_train, method='mean', resample='over'))

df_train_new = undersample_df(df_train)


h_new = np.histogram(df_train_new['label'],bins=39)[0]
plt.bar(range(1,40), h_new)
plt.title('post-sampling')
dup_amount = len(df_train_new)-len(df_train)
print('Duplicate samples: {}.\n Percentage: {}'.format(dup_amount,
                                                       dup_amount/len(df_train_new)))

df_train = df_train_new
#%% Augmentation
def train_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label
#%% Create TF dataloader
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMSIZE = (224,224,3)
BATCH_SIZE = 32

train_ds = create_tf_dataset(df_train, imsize=IMSIZE, onehot=True)

val_ds = create_tf_dataset(df_val, imsize=IMSIZE, onehot=True)


train_ds = prepare_for_training(train_ds, 
                                shuffle_buffer_size=1000,
                                batch_size=BATCH_SIZE)

val_ds = prepare_for_training(val_ds, 
                              shuffle_buffer_size=1000,
                              batch_size=BATCH_SIZE)


# augmentation
train_ds.map(train_augment, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(5):
    print(image.shape)
    print(label.shape)

I = image.numpy()
l = label.numpy()

img = I[0]
lbl = np.argmax(l[0])+1
plt.imshow(img)
