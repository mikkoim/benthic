# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:01:49 2020

@author: Mikko Impiö
"""

import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from loadbm import create_df, create_tf_dataset, prepare_for_training

#Load the ready-made splits


if platform.system() == 'Linux':
    datapath = '/home/mikko/Documents/kandi/data/IDA/Separate lists with numbering/Machine learning splits'
    img_path = '/home/mikko/Documents/kandi/data/IDA/Images/'
else:
    datapath = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Separate lists with numbering\\Machine learning splits'
    img_path = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Images\\'

def read_split(split):

    test_fname = 'test'+str(split)+'.txt'
    
    part_dat = False
    
    
    df_test = create_df(os.path.join(datapath, test_fname),
                         img_path,
                         partial_dataset=part_dat,
                         seed=123)
    
    
    return df_test

splits = []
range_ = (1,40)

y_full = []
yhat_full =[]

from metricutils import bin_classification

for s in [1,2,3,4]:
    df_test = read_split(s)
    preds = np.load(os.path.join('preds','18-01-2020_cont_colab_split{}.npy'.format(s)))
    
    y = df_test['label'].values
    yhat = np.argmax(preds,axis=1)+1
    
    
    from combine_insects import add_insect_class, add_yhat

    if False:
        
        df_test_preds = add_insect_class(df_test)
        
        # adding predictions to dataframe for insect-wise prediction
        df_test_preds = add_yhat(df_test_preds,yhat)
        dfg = df_test_preds.groupby(['label','insect'],as_index=False)['pred'].agg(lambda x:x.value_counts().index[0])
     
        y = dfg['label'].to_numpy()
        yhat = dfg['pred'].to_numpy()
        
    
    df = bin_classification(y, yhat, range_).fillna(0)
    
    split ={}
    split['df'] = df
    split['y'] = y
    split['yhat'] = yhat
    split['preds'] = preds
    
    
    splits.append(split)


dfs = []

for s in range(4):
    split = splits[s]
    dfs.append(split['df'])
    
#%%
import sklearn.metrics
from sklearn.metrics import accuracy_score


accs = [accuracy_score(split['y'], split['yhat']) for split in splits]
# lcses = [LCSE(split['y'], split['yhat']) for split in splits]

avg = 'micro'
precs = [sklearn.metrics.precision_score(split['y'], split['yhat'], average=avg) for split in splits]
recs =  [sklearn.metrics.recall_score(split['y'], split['yhat'], average=avg) for split in splits]
f1s =  [sklearn.metrics.f1_score(split['y'], split['yhat'], average=avg) for split in splits]

print(1-np.mean(accs))



#%%
    

get_metric = lambda x: pd.DataFrame([df[x] for df in dfs]).transpose()
    
    
def micro_avg(dfs, averaging='class'):
    
    getm = lambda x: np.array([df[x] for df in dfs]).transpose()
    
    if averaging=='class':
        axis=1
    elif averaging=='fold':
        axis=0
    else:
        raise Exception('invalid averaging')
    
    tp = getm('tp').sum(axis=axis)
    tn = getm('tn').sum(axis=axis)
    fp = getm('fp').sum(axis=axis)
    fn = getm('fn').sum(axis=axis)
    
    p = getm('p').sum(axis=axis)
    n = getm('n').sum(axis=axis)
    
    tpr = tp/p
    tnr = tn/n
    
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
        
    f1 = (2*tp)/(2*tp+fp+fn)
    gmean = np.sqrt(tpr*tnr)
    
    d = {'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'p': p,
        'n': n,
        'tpr': tpr,
        'tnr': tnr,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'gmean': gmean}
        
    return d

df_k = pd.DataFrame(micro_avg(dfs,'fold'))
df = df_c = pd.DataFrame(micro_avg(dfs,'class'))

df.index = range(1,40)

#%% micromacromatrix
tptot = df_c['tp'].sum()
tntot = df_c['tn'].sum()
fptot = df_c['fp'].sum()
fntot = df_c['fn'].sum()


micro_k_micro_c = (2*tptot)/(2*tptot+fptot+fntot)

macro_k_micro_c = df_c['f1'].mean()
macro_c_micro_k = df_k['f1'].mean()

macro_c_macro_k = get_metric('f1').mean(axis=1).mean()



#%% TPR, TNR, PPV, NPV

fontsize=16
marker1 = 'x'
marker2 = 'x'
range_ = range(1,40)
    
sort = df['p'].sort_values()
sizes = df['p'][sort.index]

fig, ax1 = plt.subplots()
#ax1.set_title('Primary confusion matrix metrics',  fontsize=fontsize)
ax1.bar(range_, sizes)

for i, v in enumerate(sizes):
    ax1.text(i + .5, v + sizes.max()/100, str(sizes.index[i]), color='blue', 
                                     fontsize='10') 

ax2 = ax1.twinx()  
plt.xticks([],[])
ax2.set_yticks(np.arange(0,1.1,0.1))
plt.xlabel('Class size and label',  fontsize=fontsize)

ax2.plot(range_, 
         df.loc[sort.index, 'tpr'], 
         color=[0.25, 0.5, 0.25],
         marker=marker1)

ax2.plot(range_, 
         df.loc[sort.index, 'tnr'], 
         color=[1, 0.1, 0.1],
         marker=marker1)

ax2.plot(range_, 
         df.loc[sort.index, 'ppv'], 
         color=[0, 0.5, 0], 
         linestyle='dashed',
         marker=marker2)

ax2.plot(range_, 
         df.loc[sort.index, 'npv'], 
         color=[1, 0, 0], 
         linestyle='dashed',
         marker=marker2)

ax2.plot(range_, 
         df.loc[sort.index, 'f1'], 
         color=[1, 0.3, 0.3],
         marker='+')

ax1.set_ylabel('N images', fontsize=fontsize)
ax2.set_ylabel('Metric score', fontsize=fontsize)
ax1.set_xlabel('Class label and size', fontsize=fontsize)

m = np.mean(df.loc[sort.index, 'f1'])

plt.hlines(m,0,max(range_), colors=[1, 0.3, 0.3])

plt.annotate('Macro-averaged F1 score: {:.2f}'.format(m), 
             [25,0.83],
             xytext=[20, 0.6],
             arrowprops={'arrowstyle':'->'})

plt.legend(['TPR', 'TNR', 'PPV', 'NPV', 'F1'], 
           loc='lower right',
           fontsize=fontsize-4)
ax2.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

for i in range(1,40):
    ax2.axvline(x=i, alpha=0.1)


