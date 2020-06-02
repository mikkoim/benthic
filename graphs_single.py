#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:56:16 2020

@author: mikko

"""
import numpy as np
import pandas as pd
import os
import platform
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

from loadbm import create_df

plt.style.use('default')

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


preds = np.load(os.path.join('preds','39-xx-2020.npy'))

yhat = np.argmax(preds,axis=1)+1

y = df_test['label'].to_numpy()

from combine_insects import add_insect_class, add_yhat

if False:
    
    df_test_preds = add_insect_class(df_test)
    
    # adding predictions to dataframe for insect-wise prediction
    df_test_preds = add_yhat(df_test_preds,yhat)
    dfg = df_test_preds.groupby(['label','insect'],as_index=False)['pred'].agg(lambda x:x.value_counts().index[0])
 
    y = dfg['label'].to_numpy()
    yhat = dfg['pred'].to_numpy()
    
#%%

from metricutils import bin_classification

range_ = (1,40)
df = bin_classification(y, yhat, range_).fillna(0)



#%% TPR, TNR, PPV, NPV

# assert df['tp'].values[0] == 3256

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


plt.hlines(0.8300,0,max(range_), colors=[1, 0.3, 0.3])

plt.annotate('Macro-averaged F1 score', 
             [25,0.83],
             xytext=[20, 0.6],
             arrowprops={'arrowstyle':'->'})

plt.legend(['TPR', 'TNR', 'PPV', 'NPV', 'F1'], 
           loc='lower right',
           fontsize=fontsize-4)
ax2.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

for i in range(1,40):
    ax2.axvline(x=i, alpha=0.1)

#%%
import seaborn as sns
from sklearn import metrics
C = metrics.confusion_matrix(y, yhat, normalize='true').T
fig, ax1 = plt.subplots()

ax1 = sns.heatmap(C, 
                    cmap='coolwarm',
                    annot=True, 
                    fmt='.2f', 
                    annot_kws={'size':6},
                    cbar=False,
                    xticklabels=range(1,40),
                    yticklabels=range(1,40),
                    square=True)

plt.xlabel('True')
plt.ylabel('Predicted')
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)
plt.tight_layout()


#%% Increasing order plots

# assert df['tp'].values[0] == 3256

fontsize=16
marker = 'x'
    
sort = df['f1'].sort_values()
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
         marker=marker)

ax2.plot(range_, 
         df.loc[sort.index, 'ppv'], 
         color=[0, 0.5, 0], 
         linestyle='dashed',
         marker=marker)

ax2.plot(range_, 
         df.loc[sort.index, 'f1'], 
         color=[1, 0.3, 0.3],
         marker='+')

ax2.plot(range_, 
         df.loc[sort.index, 'gmean'], 
         color=[0.4, 0.4, 0.4],
         marker='+')

ax1.set_ylabel('N images', fontsize=fontsize)
ax2.set_ylabel('Metric score', fontsize=fontsize)
ax2.set_yticks(np.arange(0,1.1,0.1))
ax1.set_xlabel('Class label and size', fontsize=fontsize)

plt.legend(['TPR', 'PPV', 'F1', 'G-mean'], loc='upper left')
ax2.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

for i in range(1,40):
    ax2.axvline(x=i, alpha=0.1)


#%% Cumulative sums

def f1_cumulative(df, i):
    c_sum = lambda col, i: df[col].iloc[:i+1].sum()

    tp = c_sum('tp',i)
    fp = c_sum('fp',i)
    fn = c_sum('fn',i)
    
    f1 = (2*tp)/(2*tp+fp+fn) 
    return f1

def tpr_cumulative(df,i):
    c_sum = lambda col, i: df[col].iloc[:i+1].sum()

    tp = c_sum('tp',i)
    fn = c_sum('fn',i)
    
    tpr = tp/(tp+fn)
    
    return tpr

def ppv_cumulative(df,i):
    c_sum = lambda col, i: df[col].iloc[:i+1].sum()

    tp = c_sum('tp',i)
    fp = c_sum('fp',i)
    
    ppv = tp/(tp+fp)
    
    return ppv
    

def calc_cumulative_score(df, sort_col, func):
    sort = df[sort_col].sort_values()
    df_sort = df.loc[sort.index]
    
    c_score = []
    for i in range(len(sort)):
        
        score = func(df_sort, i)
        
        #if np.isnan(score): score = 0
        c_score.append(score)
        
    return np.nan_to_num(np.array(c_score))


sort_col='f1'

sort = df[sort_col].sort_values() 
sizes = df['p'][sort.index]

fig, ax1 = plt.subplots()
ax1.bar(range_, sizes)

for i, v in enumerate(sizes):
    ax1.text(i + .5, v + sizes.max()/100, str(sizes.index[i]), color='blue', 
                                     fontsize='10') 
ax2 = ax1.twinx()  

score_tpr = calc_cumulative_score(df, sort_col, tpr_cumulative)
score_ppv = calc_cumulative_score(df, sort_col, ppv_cumulative)
score_f1 = calc_cumulative_score(df, sort_col, f1_cumulative)

auc_tpr = np.trapz(score_tpr, dx = 1./39)
auc_ppv = np.trapz(score_ppv, dx = 1./39)
auc_f1 = np.trapz(score_f1, dx = 1./39)

ax2.plot(range_, 
         score_tpr, 
         color=[0.25, 0.5, 0.25],
         marker=marker)

ax2.plot(range_, 
         score_ppv, 
         color=[0, 0.5, 0], 
         linestyle='dashed',
         marker=marker)

ax2.plot(range_, 
         score_f1, 
         color=[1, 0.3, 0.3],
         marker='+')

plt.xticks([],[])
ax2.set_yticks(np.arange(0,1.1,0.1))
plt.xlabel('Class size and label',  fontsize=fontsize)
plt.axis([0, max(range_)+1, 0, 1.05])
plt.title('F1 cumulative AUC: {:.4f}'.format(auc_f1), fontsize=fontsize)

ax1.set_ylabel('N images', fontsize=fontsize)
ax2.set_ylabel('Cumulative score', fontsize=fontsize)
ax2.set_yticks(np.arange(0,1.1,0.1))
ax1.set_xlabel('Class label and size', fontsize=fontsize)

plt.legend(['TPR', 
            'PPV',
            'F1'],
           loc='upper left',
           fontsize=fontsize-4)
ax2.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

for i in range(1,40):
    ax2.axvline(x=i, alpha=0.1)

