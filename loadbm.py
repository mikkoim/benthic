# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:56:14 2020

@author: Mikko Impiö
"""
import random
import os
import pandas as pd
import numpy as np

###### DATAFRAME FUNCTIONS ###############

def create_df(csvpath, 
              img_path, 
              partial_dataset=False,
              seed=123):
    
    df = pd.read_csv(csvpath,
                     delimiter=' ',
                     header=None)
    
    # take only the first 10% of datasets for testing
    random.seed(seed)
    if partial_dataset:
        
        percent = 0.05
        
        df = df.loc[random.sample(range(0,len(df)), int(percent*len(df))),:]
    # end if
        
    # clean up the splits
    df = df.iloc[:,[0,1]]
    df.columns = ["path","label"]
    df.loc[:,"path"] = df.loc[:,"path"].apply(lambda x: x.replace("\\",os.sep))
    df['path'] = df['path'].map(lambda x: img_path+x)
    return df


######### TF.DATASET FUNCTIONS ####################
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

def decode_img(img, imsize):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, imsize)

def process_path_224(filename, label):
    
    img = tf.io.read_file(filename)
    img = decode_img(img, imsize=(224,224))
    
    return img, label

def create_tf_dataset(df, imsize, onehot=True):
    """
    Creates tf.dataset from a dataframe
    """
    
    if onehot:
        lb = LabelBinarizer().fit(np.arange(1,40))
        labels = lb.transform(df.label)
    else:
        labels = df.label

    ds = tf.data.Dataset.from_tensor_slices((df.path,labels.astype(np.float32)))
    
    if imsize == (224,224,3):
        map_func = process_path_224
    else:
        map_func = None
        
    ds = ds.map(map_func, num_parallel_calls=AUTOTUNE)
    
    return ds
    
def prepare_for_training(ds, 
                         shuffle_buffer_size,
                         batch_size,
                         cache=False):
    """
    Shuffles and batches the dataset
    """

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



############# PYTHON GENERATOR FUNCTIONS #########################
    
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
