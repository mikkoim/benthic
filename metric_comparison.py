#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:04:17 2020

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

name1 = 'REF'
preds = np.load(os.path.join('preds','18-01-2020_cont_colab.npy'))

name2 = 'CB'
predsfl = np.load(os.path.join('preds','34-xx-2020.npy'))


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

#%% Basic metrics

from metricutils import calc_basic_metrics

print(name1)
acc, precision, recall, f1 = calc_basic_metrics(y, yhat)

print(name2)
acc_fl, precision_fl, recall_fl, f1_fl = calc_basic_metrics(y, yhatfl)



#%%

from metricutils import bin_classification

range_ = (1,40)
df = bin_classification(y, yhat, range_).fillna(0)
df_fl = bin_classification(y, yhatfl, range_).fillna(0)

assert np.all(df['p']==df_fl['p'])

#%% Class histogram exploration

# set height of bar
hy0 = np.histogram(y,bins=39)[0]
bars1 = hy = hy0/hy0.sum()

hyhat0 = np.histogram(yhat,bins=39)[0]
bars2 = hyhat = hyhat0/hyhat0.sum()

hyhat_fl0 = np.histogram(yhatfl,bins=39)[0]
bars3 = hyhatfl = hyhat_fl0/hyhat_fl0.sum()


# set width of bar
barWidth = 0.25
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='blue', width=barWidth, edgecolor='white', label='Groundtruth')
plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label=name1)
plt.bar(r3, bars3, color='green', width=barWidth, edgecolor='white', label=name2)
 
# Add xticks on the middle of the group bars
plt.xlabel('Class', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], [str(x) for x in range(1,40)])
 
# Create legend & Show graphic
plt.legend()
plt.show()


KL = scipy.stats.entropy(hy,hyhat)
KL_fl = scipy.stats.entropy(hy,hyhatfl)

plt.title('KL divergence {}: {:.6f}\n'
          'KL divergence {}: {:.6f}'.format(name1, KL,
                                            name2, KL_fl))

#%% F1 vs class size



def score_vs_size(metric, sortcol='p'):
    
    sort = df[sortcol].sort_values()
    sizes = df['p'][sort.index]
    
    fig, ax1 = plt.subplots()
    ax1.set_title('{} vs class size'.format(metric))
    ax1.bar(range(1,40), sizes)
    
    for i, v in enumerate(sizes):
        ax1.text(i + .75, v + sizes.max()/100, str(sizes.index[i]), color='blue', 
                                         fontsize='10') 
    
    ax2 = ax1.twinx()  
    plt.xticks([],[])
    plt.xlabel('Class size and label')
    
    ax2.plot(range(1,40), df_fl.loc[sort.index, metric], color='r',)
    ax2.plot(range(1,40),    df.loc[sort.index, metric], color='b', linestyle='dashed')
    
    plt.legend([name2, name1])
    
score_vs_size('tpr', 'p')
score_vs_size('ppv', 'p')
score_vs_size('f1', 'p')

score_vs_size('tpr', 'tpr')
score_vs_size('ppv', 'ppv')
score_vs_size('f1', 'f1')
#%% Cumulative f1 and mcc vs class size

def f1_cumulative(df, i):
    c_sum = lambda col, i: df[col].iloc[:i+1].sum()

    tp = c_sum('tp',i)
    fp = c_sum('fp',i)
    fn = c_sum('fn',i)
    
    f1 = (2*tp)/(2*tp+fp+fn) 
    return f1

def mcc_cumulative(df, i):
    c_sum = lambda col, i: df[col].iloc[:i+1].sum()

    tp = c_sum('tp',i)
    fp = c_sum('fp',i)
    tn = c_sum('tn',i)
    fn = c_sum('fn',i)
    
    mcc = calc_mcc(tp, fp, tn, fn)
    return mcc

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

def plot_cumulative_score(func, title, sort_col='p'):

    score = calc_cumulative_score(df, sort_col, func)
    score_fl = calc_cumulative_score(df_fl, sort_col, func)
    
    sort = df[sort_col].sort_values()
    fig, ax1 = plt.subplots()
    ax1.bar(range(1,40), df.loc[sort.index, 'p'])
    
    ax2 = ax1.twinx() 
    for i, v in enumerate(sort):
        ax1.text(i + .75, v + sort.max()/100, str(sort.index[i]), 
                                         fontsize='10', color='b') 
    
    ax2.plot(range(1,40), score, color='b', linestyle='dashed')
    ax2.plot(range(1,40), score_fl, color='r')
    
    plt.xticks([],[])
    
    plt.legend([name1+' '+title, name2+' '+title])
    
    auc = np.trapz(score, dx = 1./39)
    auc_fl = np.trapz(score_fl, dx = 1./39)
    
    plt.title('Cumulative {title} vs class size. \n'
              '{name1} {title}: {:.4f}, {name2} {title}: {:.4f}\n'.format(auc,
                                                                          auc_fl, 
                                                                          title=title,
                                                                          name1=name1,
                                                                          name2=name2))
    

plot_cumulative_score(tpr_cumulative, 'TPR')
plot_cumulative_score(ppv_cumulative, 'Precision')
plot_cumulative_score(f1_cumulative, 'F1')

#%% Class size vs false sum

falses    = (df['fp']+df['fn'])/df['p']
falses_fl = (df_fl['fp']+df_fl['fn'])/df_fl['p']

sort = df['p'].sort_values()
fig, ax1 = plt.subplots()
ax1.set_title('Class size vs false sum')
ax1.bar(range(1,40), df.loc[sort.index, 'p'], color='b')
for i, v in enumerate(sort):
    ax1.text(i + .75, v + sort.max()/100, str(sort.index[i]), color='blue', 
                                     fontsize='10') 

ax2 = ax1.twinx()  
ax2.plot(range(1,40), falses.loc[sort.index].values, color='r', linestyle='dashed')
ax2.plot(range(1,40), falses_fl.loc[sort.index].values, color='r')

plt.legend(['{} falses'.format(name1), '{} falses'.format(name2)])

#%% bar plots

def barplot(df, ax=None):
    width = 0.9
    sort = df['p'].sort_values()
    
    tpr = df['tpr'].values
    fpr = 1-df['tnr'].values
    fnr = 1-tpr
    
    tpr = tpr[sort.index-1]
    fpr = fpr[sort.index-1]
    fnr = fnr[sort.index-1]
    
    ind = np.arange(39)
    
    if not ax:
        fig, ax = plt.subplots()
        
    p1 = ax.bar(ind, tpr, width=width, color='green')
    p2 = ax.bar(ind, fpr, width=width, bottom=tpr, color='orange')
    p3 = ax.bar(ind, -fnr, width=width, color='red')
    plt.xticks(range(39),sort.index)
    plt.axis([-0.1,40,-1.1,1.1])
    
barplot(df)
plt.title('True positive detection distribution\n{}'.format(name1))
barplot(df_fl)
plt.title('True positive detection distribution\n{}'.format(name2))
