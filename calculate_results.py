# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:43:47 2020

@author: Mikko Impiö
"""

import numpy as np
import pandas as pd
import os
import platform
import matplotlib.pyplot as plt
import scipy

from loadbm import create_df


# load testset and preds

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


modelfiles = ['18-01-2020_cont_colab.npy',
              '39-xx-2020.npy',
              '34-xx-2020.npy',
              '38-xx-2020.npy',
              '35-xx-2020.npy',
              '36-xx-2020.npy',
              '29-02-2020_cont_colab.npy',
              '10-02-2020_cont_colab.npy']

modelnames = ['Reference',
              'wCEinv',
              'wCEnorm',
              'FLg2',
              'FLg4',
              'CB',
              'Overample_cont',
              'Augment']

p = {}
for i in modelnames: 
    p[i] = {}
    
for file, name in zip(modelfiles, modelnames):
    preds = np.load(os.path.join('preds',file))
    p[name]['preds'] = preds
    p[name]['yhat'] = np.argmax(preds,axis=1)+1
    
y = df_test['label'].to_numpy()

#%% inits


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer().fit(range(1,40))
Y_test = lb.transform(y)
n_classes = 39


#%% 
from sklearn.metrics import accuracy_score, f1_score

d = {'model': [],
     'acc': [],
     'lcse': [],
     'f1': [],
     'AP': []}

for model in modelnames:
    preds = p[model]['preds']
    yhat = p[model]['yhat']
    
    # Accuracy, LCSE and F1
    acc = accuracy_score(y, yhat)
    lcse = LCSE(y, yhat)
    f1 = f1_score(y, yhat, average='macro')
    
    # PCAUC
    y_score = preds
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    
    AP = average_precision["micro"]
    
    d['model'].append(model)
    d['acc'].append(acc)
    d['lcse'].append(lcse)
    d['f1'].append(f1)
    d['AP'].append(AP)
    
results = pd.DataFrame(d)

    
    