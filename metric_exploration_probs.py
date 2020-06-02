# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:58:49 2020

@author: Mikko Impiö
"""

import numpy as np
import pandas as pd
import os
import platform
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

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


preds = np.load('10-02-2020_cont_colab.npy')
predsfl = np.load('22-01-2020_cont_colab.npy')

yhat = np.argmax(preds,axis=1)+1
yhatfl = np.argmax(predsfl,axis=1)+1

from sklearn.preprocessing import LabelBinarizer

y = df_test['label'].to_numpy()

lb = LabelBinarizer().fit(range(1,40))

yhot = lb.transform(y)
#%%
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(39):
    fpr[i], tpr[i], _ = roc_curve(yhot[:, i], preds[:, i])
    if np.isnan(tpr[i]).any() or np.isnan(fpr[i]).any():
        fpr[i] = tpr[i] = np.zeros(39)
        
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(yhot.ravel(), preds.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(39)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(39):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
mean_tpr /= 39

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

for i in range(39):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right", fontsize='xx-small')
plt.show()

#%%

x = preds[4,:]
