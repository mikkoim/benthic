# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:29:58 2020

@author: Mikko Impiö
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import platform

#Load the ready-made splits

if platform.system() == 'Linux':
    datapath = '/home/mikko/Documents/kandi/data/IDA/Separate lists with numbering/Machine learning splits'
    img_path = '/home/mikko/Documents/kandi/data/IDA/Images/'
else:
    datapath = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Separate lists with numbering\\Machine learning splits'
    img_path = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Images\\'

split = 1

test_fname = 'test'+str(split)+'.txt'

df_load = lambda fname: pd.read_csv(os.path.join(datapath,fname),
                                    delimiter=' ',
                                    header=None)

df_test = df_load(test_fname)

# take only the first 10% of datasets for testing
import random
random.seed(123)
partial_dataset = True
if partial_dataset:
    
    percent = 0.05
    
    d = lambda df: df.loc[random.sample(range(0,len(df)), int(percent*len(df))),:]
    
    df_test = d(df_test)

# clean up the splits
def df_preprocess(df):
    df = df.iloc[:,[0,1]]
    df.columns = ["path","label"]
    df.loc[:,"path"] = df.loc[:,"path"].apply(lambda x: x.replace("\\",os.sep))
    df['path'] = df['path'].map(lambda x: img_path+x)
    return df

# The resulting dataframes of the splits

df_test = df_preprocess(df_test)

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

test_ds = make_dataset(df_test).map(process_path, num_parallel_calls=AUTOTUNE)
##

#%% Evaluation

from tensorflow.keras.models import load_model
model = load_model('09-01-2020.h5')

test_ds = test_ds.batch(BATCH_SIZE)

preds = model2.predict(test_ds, verbose=1)

model.evaluate(test_ds)

#%%

take_test = test_ds.take(8)

it = iter(take_test)

for t in it:
    print(t[1])


loss = tf.keras.losses.categorical_crossentropy