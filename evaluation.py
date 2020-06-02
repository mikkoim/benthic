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

from loadbm import create_df, create_tf_dataset


#Load the ready-made splits

if platform.system() == 'Linux':
    datapath = '/home/mikko/Documents/kandi/data/IDA/Separate lists with numbering/Machine learning splits'
    img_path = '/home/mikko/Documents/kandi/data/IDA/Images/'
else:
    datapath = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Separate lists with numbering\\Machine learning splits'
    img_path = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Images\\'

split = 1
test_fname = 'test'+str(split)+'.txt'

part_dat = False

df_test = create_df(os.path.join(datapath, test_fname),
                     img_path,
                     partial_dataset=part_dat,
                     seed=123)

#%% Create TF dataloader

IMSIZE = (224,224,3)
BATCH_SIZE = 32

test_ds = create_tf_dataset(df_test, imsize=IMSIZE, onehot=True)
test_ds = test_ds.batch(BATCH_SIZE)
##

#%% Evaluation

modelpth = 'D:\\Users\\Mikko Impiö\\kandi\\models'

from tensorflow.keras.models import load_model
model = load_model(os.path.join(modelpth,'09-02-2020_cont_colab.h5'))

preds = model.predict(test_ds, verbose=True)

yhat = np.argmax(preds,axis=1)+1

y_test = df_test['label']

#%% Insect combine

from combine_insects import add_insect_class, add_yhat

df_test_preds = add_insect_class(df_test)

# adding predictions to dataframe for insect-wise prediction
df_test_preds = add_yhat(df_test_preds,yhat)

dfg = df_test_preds.groupby(['label','insect'],as_index=False)['pred'].agg(lambda x:x.value_counts().index[0])

acc = np.sum(yhat==y_test)/len(y_test)
print('Image accuracy: {:.4f}'.format(acc))

acc_g = np.sum(dfg['pred']==dfg['label'])/len(dfg)
print('Aggregate accuracy: {:.4f}'.format(acc_g))
