# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 22:00:11 2020

@author: Mikko Impiö
"""

import numpy as np
import pandas as pd
import os
import platform
import matplotlib.pyplot as plt
import tensorflow as tf

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

from loadbm import create_df
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


#%% Generator train
from sklearn.preprocessing import LabelBinarizer
from loadbm import data_generator

lb = LabelBinarizer()
lb.fit(np.arange(1,40))

BATCH_SIZE = 8

dgtrain = data_generator(BATCH_SIZE, df_train, lb, israndom=True) 
dgval = data_generator(BATCH_SIZE, df_val, lb, israndom=True) 


#%% NN Model

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def get_pretrained(imsize=(224, 224, 3), classes=39):
        base_model = InceptionV3(input_shape = imsize, 
                                 weights='imagenet', 
                                 include_top=False)
        
        base_model.trainable = True
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model

model = get_pretrained()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics=['accuracy'])

#%% Training and logging
from tensorflow.keras.callbacks import CSVLogger
logdir = 'C:\\Users\\Mikko Impiö\\Google Drive\\koulu_honmia\\kandi19\\logs'

csv_logger = CSVLogger(os.path.join(logdir,'10-01-2020_2.log'),append=True)

tr_steps = len(df_train)//BATCH_SIZE
val_steps = len(df_val)//BATCH_SIZE

model.fit_generator(dgtrain, 
                    validation_data= dgval, 
                    steps_per_epoch= tr_steps, 
                    epochs = 5,
                    validation_steps = val_steps,
                    callbacks=[csv_logger])