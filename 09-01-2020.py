# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:02:22 2020

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

#%%
#check the contents of the dataset
N_TAXA = 39
len(np.unique(df_train['label']))
len(np.unique(df_test['label']))
len(np.unique(df_val['label']))

# = df_test.hist(bins=N_TAXA)

#%% Create TF dataloader
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer().fit(np.arange(1,40))

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(filename, label):
    
    img = tf.io.read_file(filename)
    img = decode_img(img)
    
    return img, label

make_dataset = lambda df: tf.data.Dataset.from_tensor_slices((df.path,df.label))

train_ds = make_dataset(df_train).map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = make_dataset(df_test).map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = make_dataset(df_val).map(process_path, num_parallel_calls=AUTOTUNE)


##

def prepare_for_training(ds, 
                         cache=False, 
                         shuffle_buffer_size=1000,
                         batch_size=BATCH_SIZE):

  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(batch_size)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

train_ds = prepare_for_training(train_ds, shuffle_buffer_size=len(df_train))
val_ds = prepare_for_training(val_ds, shuffle_buffer_size=len(df_val))

for image, label in train_ds.take(5):
    print(image.shape)
    print(label.shape)


#%% NN Model

from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def get_pretrained(imsize=(IMG_HEIGHT, IMG_WIDTH, 3), classes=39):
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
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
logdir = 'C:\\Users\\Mikko Impiö\\Google Drive\\koulu_honmia\\kandi19\\logs'

tb = TensorBoard(log_dir=logdir, histogram_freq=1)

csv_logger = CSVLogger(os.path.join(logdir,'09-01-2020.log'))

tr_steps = len(df_train)//BATCH_SIZE
val_steps = len(df_val)//BATCH_SIZE

model.fit_generator(train_ds, 
                    validation_data= val_ds, 
                    steps_per_epoch= tr_steps, 
                    epochs = 2,
                    validation_steps = val_steps,
                    callbacks=[csv_logger, tb])

#%% Inference

test_ds = test_ds.batch(BATCH_SIZE)
model.evaluate(test_ds)
