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

df_load = lambda fname: pd.read_csv(os.path.join(datapath,fname),
                                    delimiter=' ',
                                    header=None)

df_train = df_load(train_fname)
df_test = df_load(test_fname)
df_val =df_load(val_fname)

# take only the first 10% of datasets for testing
import random
random.seed(123)
partial_dataset = True
if partial_dataset:
    
    percent = 0.05
    
    d = lambda df: df.loc[random.sample(range(0,len(df)), int(percent*len(df))),:]
    
    df_train = d(df_train)
    df_test = d(df_test)
    df_val = d(df_val)

# clean up the splits
def df_preprocess(df):
    df = df.iloc[:,[0,1]]
    df.columns = ["path","label"]
    df.loc[:,"path"] = df.loc[:,"path"].apply(lambda x: x.replace("\\",os.sep))
    df['path'] = df['path'].map(lambda x: img_path+x)
    return df

# The resulting dataframes of the splits

df_train = df_preprocess(df_train)
df_test = df_preprocess(df_test)
df_val = df_preprocess(df_val)



#%% Generator train
from sklearn.preprocessing import LabelBinarizer
from PIL import Image

def shuffle(files,labels):
    
    mapIndexPosition = list(zip(files, labels))
    random.shuffle(mapIndexPosition)
    files, labels = zip(*mapIndexPosition)
    
    return files, labels

def data_generator(batch_size, df, lb, imsize=(224,224,3), israndom=True):
    
    files = df.loc[:,"path"].tolist()
    labels = lb.transform(df.loc[:,"label"].tolist())
    
    if israndom:
        files, labels = shuffle(files, labels)
        
    j = 0
    while True:
        inputs = []
        targets = []
        for i in range(batch_size):
            
            ind = j+i
                
            name = files[ind]
            
            img = Image.open(name).resize(imsize[0:2])
            
            img = np.asarray(img)
            img = (img - np.mean(img))/np.std(img)


            label = labels[ind]
            
            inputs.append(img)
            targets.append(label)
            
        j = j + i
        
        if j + batch_size >= len(files) and israndom:
            
            files, labels = shuffle(files, labels)
            j = 0
            
        yield np.asarray(inputs), np.asarray(targets)

lb = LabelBinarizer()
lb.fit(np.arange(1,40))

BATCH_SIZE = 8
imsize = (224,224,3)

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

csv_logger = CSVLogger(os.path.join(logdir,'10-01-2020.log'))

tr_steps = len(df_train)//BATCH_SIZE
val_steps = len(df_val)//BATCH_SIZE

model.fit_generator(dgtrain, 
                    validation_data= dgval, 
                    steps_per_epoch= tr_steps, 
                    epochs = 5,
                    validation_steps = val_steps,
                    callbacks=[csv_logger])