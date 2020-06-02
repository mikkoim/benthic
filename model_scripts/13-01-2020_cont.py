# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 01:51:11 2020

@author: Mikko Impiö
"""

import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import tensorflow as tf

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


#%% Create TF dataloader
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMSIZE = (224,224,3)
BATCH_SIZE = 8

train_ds = create_tf_dataset(df_train, imsize=IMSIZE, onehot=True)

val_ds = create_tf_dataset(df_val, imsize=IMSIZE, onehot=True)


train_ds = prepare_for_training(train_ds, 
                                shuffle_buffer_size=1000,
                                batch_size=BATCH_SIZE)

val_ds = prepare_for_training(val_ds, 
                              shuffle_buffer_size=1000,
                              batch_size=BATCH_SIZE)

for image, label in train_ds.take(5):
    print(image.shape)
    print(label.shape)

#%%
    
    
modelpth = 'D:\\Users\\Mikko Impiö\\kandi\\models'

from tensorflow.keras.models import load_model
model = load_model(os.path.join(modelpth,'13-01-2020_6epoch.h5'))

from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.0001) #original 0.001

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy',
                  metrics=['accuracy'])

#%%

from tensorflow.keras.callbacks import CSVLogger
logdir = 'C:\\Users\\Mikko Impiö\\Google Drive\\koulu_honmia\\kandi19\\logs'

csv_logger = CSVLogger(os.path.join(logdir,'13-01-2020_cont.log'),append=True)

tr_steps = len(df_train)//BATCH_SIZE
val_steps = len(df_val)//BATCH_SIZE

model.fit_generator(train_ds, 
                    validation_data= val_ds, 
                    steps_per_epoch= tr_steps, 
                    epochs = 30,
                    validation_steps = val_steps,
                    callbacks=[csv_logger])
