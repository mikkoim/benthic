# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:07:14 2020

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

#%% Create TF dataloader
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMSIZE = (224,224,3)
BATCH_SIZE = 8

train_ds = create_tf_dataset(df_train, imsize=IMSIZE, onehot=True)

val_ds = create_tf_dataset(df_val, imsize=IMSIZE, onehot=True)


train_ds = prepare_for_training(train_ds, 
                                shuffle_buffer_size=len(df_train),
                                batch_size=BATCH_SIZE)

val_ds = prepare_for_training(val_ds, 
                              shuffle_buffer_size=len(df_val),
                              batch_size=BATCH_SIZE)

for image, label in train_ds.take(5):
    print(image.shape)
    print(label.shape)


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

csv_logger = CSVLogger(os.path.join(logdir,'11-01-2020_2.log'),append=True)

tr_steps = len(df_train)//BATCH_SIZE
val_steps = len(df_val)//BATCH_SIZE

model.fit_generator(train_ds, 
                    validation_data= val_ds, 
                    steps_per_epoch= tr_steps, 
                    epochs = 30,
                    validation_steps = val_steps,
                    callbacks=[csv_logger])

#%% Inference

test_ds = test_ds.batch(BATCH_SIZE)
model.evaluate(test_ds)
