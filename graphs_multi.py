#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:56:16 2020

@author: mikko

"""

#%%

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

df1name = 'Reference'
df2name = r'Ref + oversampled'

preds = np.load(os.path.join('preds','18-01-2020_cont_colab.npy'))

predsfl = np.load(os.path.join('preds','10-02-2020_cont_colab.npy'))


yhat = np.argmax(preds,axis=1)+1
yhatfl = np.argmax(predsfl,axis=1)+1

y = df_test['label'].to_numpy()

from combine_insects import add_insect_class, add_yhat

if False:
    
    df_test_preds = add_insect_class(df_test)
    
    # adding predictions to dataframe for insect-wise prediction
    df_test_preds = add_yhat(df_test_preds,yhat)
    dfg = df_test_preds.groupby(['label','insect'],as_index=False)['pred'].agg(lambda x:x.value_counts().index[0])

    
    df_test_preds_fl = add_yhat(df_test_preds,yhatfl)
    dfg_fl = df_test_preds_fl.groupby(['label','insect'],as_index=False)['pred'].agg(lambda x:x.value_counts().index[0])
    
    y = dfg['label'].to_numpy()
    yhat = dfg['pred'].to_numpy()
    yhatfl = dfg_fl['pred'].to_numpy()

#%%

from metricutils import bin_classification

range_ = (1,40)
df = bin_classification(y, yhat, range_).fillna(0)
df_fl = bin_classification(y, yhatfl, range_).fillna(0)

assert np.all(df['p']==df_fl['p'])


#%% TPR, TNR, PPV, NPV

sortcol = 'f1'
metric = 'f1'
metricname = 'F1'

# assert df['tp'].values[0] == 3256

fontsize=16
marker1 = 'x'
marker2 = 'x'
range_ = range(1,40)
    
sort = df[sortcol].sort_values()
sizes = df['p'][sort.index]

fig, ax1 = plt.subplots()
ax1.bar(range_, sizes)

for i, v in enumerate(sizes):
    ax1.text(i + .45, v + sizes.max()/100, str(sizes.index[i]), color='blue', 
                                     fontsize='9') 

ax2 = ax1.twinx()  
plt.xticks([],[])
plt.xlabel('Class size and label',  fontsize=fontsize)

ax2.plot(range_, 
         df.loc[sort.index, metric], 
         color='b',
         linestyle='dashed',
         marker='+')

ax2.plot(range_, 
         df_fl.loc[sort.index, metric], 
         color='r',
         linestyle='solid',
         marker='.')



ax1.set_ylabel('N images', fontsize=fontsize)
ax2.set_ylabel('Metric score', fontsize=fontsize)
ax1.set_xlabel('Class label and size', fontsize=fontsize)

m    = np.mean(df.loc[sort.index, metric])
m_fl = np.mean(df_fl.loc[sort.index, metric])

plt.title("{} macro {}: {:.2f} \n {} macro {}: {:.2f}".format(df1name,
                                                         metricname,
                                                         m,
                                                         df2name,
                                                         metricname,
                                                         m_fl),
          fontsize=fontsize-2)

plt.hlines(m,0,max(range_), 
           colors='b', 
           linestyle='dashed')

plt.hlines(m_fl,0,max(range_), 
           colors='r', 
           linestyle='solid')

plt.legend(['{} {}'.format(df1name, metricname), 
            '{} {}'.format(df2name, metricname)], 
           loc='lower right',
           fontsize=fontsize-4)

ax1.set_axisbelow(True)
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


sortcol = 'f1'
metric = 'f1'
metricname = 'F1'

sort1 = df[sortcol].sort_values() 
sort2 = df_fl[sortcol].sort_values() 

sizes1 = df['p'][sort1.index]
sizes2 = df_fl['p'][sort2.index]

def plot_cumulative(df, name, sortcol, ax, color):

    score_tpr = calc_cumulative_score(df, sortcol, tpr_cumulative)
    score_ppv = calc_cumulative_score(df, sortcol, ppv_cumulative)
    score_f1 = calc_cumulative_score(df, sortcol, f1_cumulative)
    
    auc_tpr = np.trapz(score_tpr, dx = 1./39)
    auc_ppv = np.trapz(score_ppv, dx = 1./39)
    auc_f1 = np.trapz(score_f1, dx = 1./39)
    
    r = {'tpr': (score_tpr, auc_tpr),
         'ppv': (score_ppv, auc_ppv),
         'f1':  (score_f1, auc_f1)}
    
    ax.plot(range_, 
             score_tpr, 
             color=color,
             linestyle='dashed',
             label="{} TPR".format(name))
    
    ax.plot(range_, 
             score_ppv, 
             color=color, 
             linestyle='dashdot',
             label="{} PPV".format(name))
    
    ax.plot(range_, 
             score_f1, 
             color=color,
             marker='+',
             label="{} F1".format(name))
    
    ax.fill_between(range_, score_ppv, score_tpr, alpha=0.3, color=color)
    
    return r

fig, ax1 = plt.subplots()
scores      = plot_cumulative(df, 
                              df1name, 
                              sortcol, 
                              ax1, 
                              color='b')

scores_fl   = plot_cumulative(df_fl, 
                              df2name, 
                              sortcol, 
                              ax1, 
                              color='r')

plt.xticks([],[])
plt.axis([0, max(range_)+1, 0, 1.05])

plt.title("{} {} AUC: {:.2f} \n {} {} AUC: {:.2f}".format(df1name,
                                                             metricname,
                                                             scores['f1'][1],
                                                             df2name,
                                                             metricname,
                                                             scores_fl['f1'][1]),
          fontsize=fontsize-2)


ax1.set_ylabel('Cumulative score', fontsize=fontsize)

plt.legend(loc='lower right',
           fontsize=fontsize-4)
plt.gca().yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)

#%%

sortcol = 'p'
metric = 'f1'
metricname = 'F1'

# assert df['tp'].values[0] == 3256

fontsize=16
marker1 = 'x'
marker2 = 'x'
range_ = range(1,40)
    
sort = df[sortcol].sort_values()
sizes = df['p'][sort.index]

fig, ax1 = plt.subplots()
ax1.bar(range_, sizes)
ax2 = ax1.twinx()  

sorted_metrics1 = df.loc[sort.index, metric]
sorted_metrics2 = df_fl.loc[sort.index, metric]

for i, v in enumerate(sizes):
    ax1.text(i + .45, v + sizes.max()/100, str(sizes.index[i]), color='blue', 
                                     fontsize='9') 
    
    ar_base =  sorted_metrics1.values[i]
    ar_end = sorted_metrics2.values[i]
    
    dy = ar_end-ar_base
    
    if dy>0: 
        ar_color = 'g' 
    else: 
        ar_color = 'r'
    
    ax2.bar(i+1, dy, width=0.6, bottom=ar_base, color=ar_color,zorder=-1)


plt.xticks([],[])
plt.xlabel('Class size and label',  fontsize=fontsize)

ax2.scatter(range_, 
          df.loc[sort.index, metric], 
          color='b',
          marker='+')

ax2.scatter(range_, 
          df_fl.loc[sort.index, metric], 
          color='r',
          marker='.')



ax1.set_ylabel('N images', fontsize=fontsize)
ax2.set_ylabel('Metric score', fontsize=fontsize)
ax1.set_xlabel('Class label and size', fontsize=fontsize)

m    = np.mean(df.loc[sort.index, metric])
m_fl = np.mean(df_fl.loc[sort.index, metric])

plt.title("{} macro {}: {:.2f} \n {} macro {}: {:.2f}".format(df1name,
                                                         metricname,
                                                         m,
                                                         df2name,
                                                         metricname,
                                                         m_fl),
          fontsize=fontsize-2)

plt.hlines(m,0,max(range_), 
           colors='b', 
           linestyle='dashed')

plt.hlines(m_fl,0,max(range_), 
           colors='r', 
           linestyle='solid')

plt.legend(['{} {}'.format(df1name, metricname), 
            '{} {}'.format(df2name, metricname)], 
           loc='lower right',
           fontsize=fontsize-4)

ax1.set_axisbelow(True)
ax2.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
ax2.xaxis.grid(color='gray', linestyle='dotted', alpha=0.2)

for i in range(1,40):
    ax2.axvline(x=i, alpha=0.1)