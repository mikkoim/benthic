# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:14:45 2020

@author: Mikko Impiö
"""

import numpy as np
import pandas as pd
import os
import platform
from PIL import Image

from loadbm import create_df, create_tf_dataset, prepare_for_training

#Load the ready-made splits
if platform.system() == 'Linux':
    datapath = '/home/mikko/Documents/kandi/data/IDA/Separate lists with numbering/Machine learning splits'
    img_path = '/home/mikko/Documents/kandi/data/IDA/Images/'
else:
    datapath = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Separate lists with numbering\\Machine learning splits'
    img_path = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Images\\'

split = 1
test_fname = 'test'+str(split)+'.txt'

part_dat = True

df_test = create_df(os.path.join(datapath, test_fname),
                     img_path,
                     partial_dataset=part_dat,
                     seed=123)


#%% Loading in memory
import tensorflow as tf

def create_tf_img(fname, imsize):
    img = tf.io.read_file(fname)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, imsize[0:-1])


def load_images(df, img_path, imsize=(224,224,3)):
    imglist = df.loc[:,"path"].tolist()
    X = np.zeros([len(imglist),imsize[0],imsize[1],imsize[2]])
    for i, fname in enumerate(imglist):
        
        img = create_tf_img(fname, imsize)
        
        X[i,:,:,:] = img
        
        print("{}/{}".format(i,len(imglist)))
    y = np.asarray(df.loc[:,"label"].tolist())
    return X, y
       
imsize = (224,224,3)
X_test, y_test = load_images(df_test, img_path, imsize)

#%%

from tensorflow.keras.models import load_model
model = load_model('10-01-2020-30epoch.h5')

preds = model.predict(X_test, verbose=True)

yhat = np.argmax(preds,axis=1)+1

np.sum(yhat==y_test)/len(yhat)
