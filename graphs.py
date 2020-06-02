#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:56:16 2020

@author: mikko

"""

#%%
names = ['1-Agapetus sp.',
         '2-Ameletus inopinatus',
         '3-Amphinemura borealis',
         '4-Baetis rhodani',
         '5-Baetis vernus group',
         '6-Capnopsis schilleri',
         '7-Diura sp.',
         '8-Elmis aenea',
         '9-Ephemerella aurivillii',
         '10-Ephemerella mucronata',
         '11-Heptagenia sulphurea',
         '12-Hydraena sp.',
         '13-Hydropsyche pellucidula',
         '14-Hydropsyche saxonica',
         '15-Hydropsyche siltalai',
         '16-Isoperla sp.',
         '17-Kageronia fuscogrisea',
         '18-Lepidostoma hirtum',
         '19-Leptophlebia sp.',
         '20-Leuctra nigra',
         '21-Leuctra sp.',
         '22-Limnius volckmari',
         '23-Micrasema gelidum',
         '24-Micrasema setiferum',
         '25-Nemoura cinerea',
         '26-Nemoura sp.',
         '27-Neureclipsis bimaculata',
         '28-Oulimnius tuberculatus',
         '29-Oxyethira sp.',
         '30-Plectrocnemia sp.',
         '31-Polycentropus flavomaculatus',
         '32-Polycentropus irroratus',
         '33-Protonemura sp.',
         '35-Sialis sp.',
         '36-Rhyacophila nubila',
         '36-Silo pallipes',
         '37-Simuliidae',
         '38-Sphaerium sp.',
         '39-Taeniopteryx nebulosa']

# 1-Agapetus sp.
# 2-Ameletus inopinatus
# 3-Amphinemura borealis
# 4-Baetis rhodani
# 5-Baetis vernus group
# 6-Capnopsis schilleri
# 7-Diura sp.
# 8-Elmis aenea
# 9-Ephemerella aurivillii
# 10-Ephemerella mucronata
# 11-Heptagenia sulphurea
# 12-Hydraena sp.
# 13-Hydropsyche pellucidula
# 14-Hydropsyche saxonica
# 15-Hydropsyche siltalai
# 16-Isoperla sp.
# 17-Kageronia fuscogrisea
# 18-Lepidostoma hirtum
# 19-Leptophlebia sp.
# 20-Leuctra nigra
# 21-Leuctra sp.
# 22-Limnius volckmari
# 23-Micrasema gelidum
# 24-Micrasema setiferum
# 25-Nemoura cinerea
# 26-Nemoura sp.
# 27-Neureclipsis bimaculata
# 28-Oulimnius tuberculatus
# 29-Oxyethira sp.
# 30-Plectrocnemia sp.
# 31-Polycentropus flavomaculatus
# 32-Polycentropus irroratus
# 33-Protonemura sp.
# 35-Sialis sp.
# 36-Rhyacophila nubila
# 36-Silo pallipes
# 37-Simuliidae
# 38-Sphaerium sp.
# 39-Taeniopteryx nebulosa

#%%

import matplotlib.pyplot as plt
import numpy as np


e = 1e-7
q = np.arange(0+e,2,0.001)

L = -np.log(q)
FL2 = -(1-q)**2*np.log(q)
FL4 = -(1-q)**4*np.log(q)

plt.plot(q,L)
plt.plot(q,FL2)
plt.plot(q,FL4)

plt.axis([0,1,0,5])
plt.ylabel('Loss', fontsize=14)
plt.xlabel('$q_c(x_i)$ value', fontsize=14)
plt.gca().set_aspect(0.1)

plt.legend(['Cross-entropy loss',
            'Focal loss with $\gamma$ = 2',
            'Focal loss with $\gamma$ = 4'],
            fontsize=12)
#plt.savefig('lossgraph.png', dpi=300)

#%%

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

def read_split_to_dfs(split):

    train_fname = 'train'+str(split)+'.txt'
    test_fname = 'test'+str(split)+'.txt'
    val_fname = 'val'+str(split)+'.txt'
    
    part_dat = False
    
    df_train = create_df(os.path.join(datapath, train_fname),
                         img_path,
                         partial_dataset=part_dat,
                         seed=123)
    
    df_test = create_df(os.path.join(datapath, test_fname),
                         img_path,
                         partial_dataset=part_dat,
                         seed=123)
    
    df_val = create_df(os.path.join(datapath, val_fname),
                         img_path,
                         partial_dataset=part_dat,
                         seed=123)
    
    return df_train, df_test, df_val

splits = {}
splits['train'] = []
splits['test'] = []
splits['val'] = []
for split in range(1,11):
    df_train, df_test, df_val = read_split_to_dfs(split)
    
    splits['train'].append(df_train['label'].values)
    splits['test'].append(df_test['label'].values)
    splits['val'].append(df_val['label'].values)
    
df_test,df_train,df_val = read_split_to_dfs(1)

df = pd.concat([df_test,df_train,df_val])
y = df['label'].values



#%%

href = np.histogram(y,bins=39)[0]
c = np.arange(1,40)
plt.bar(c,href, color='red')
plt.title('Full data histogram')

#%% 

sort = np.sort(href)
ind = np.argsort(href)

fig, ax1 = plt.subplots()
ax1.bar(range(1,40), sort, color='red')

for i, v in enumerate(sort):
    ax1.text(i + .75, v + sort.max()/100, str(ind[i]+1), color='red', 
                                     fontsize='10') 
    
fontsize = 12
plt.xticks(np.arange(1,40), ind+1, fontsize=fontsize)

plt.yticks(fontsize=fontsize)
plt.ylabel('N images', fontsize=16)
plt.xlabel('Class label', fontsize=16)

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

ax1.set_ylabel('N images over all folds', fontsize=fontsize)
ax2.set_ylabel('Micro-average score', fontsize=fontsize)
ax1.set_xlabel('Class label and size', fontsize=fontsize)

m    = np.mean(df.loc[sort.index, 'f1'])
plt.hlines(m,0,max(range_), colors=[1, 0.3, 0.3])
plt.annotate('Macro-averaged F1 score: {:.2f}'.format(m), 
             [25,m],
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

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# plt.savefig('ref_confmat.png', dpi=300)

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


m    = np.mean(df.loc[sort.index, 'f1'])
plt.hlines(m,0,max(range_), colors=[1, 0.3, 0.3])
plt.annotate('Macro-averaged F1 score: {:.2f}'.format(m), 
             [25,m],
             xytext=[20, 0.6],
             arrowprops={'arrowstyle':'->'})

ax1.set_ylabel('N images over all folds', fontsize=fontsize)
ax2.set_ylabel('Micro-average score', fontsize=fontsize)
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
plt.title('F1 cumulative AUC: {:.3f}'.format(auc_f1), fontsize=fontsize)

ax1.set_ylabel('N images over all folds', fontsize=fontsize)
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

#%%
fontsize=16
# set width of bar
barWidth = 0.33
 
# set height of bar
hy0 = np.histogram(y,bins=39)[0]
bars1 = hy = hy0/hy0.sum()

hyhat0 = np.histogram(yhat,bins=39)[0]
bars2 = hyhat = hyhat0/hyhat0.sum()
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='red', width=barWidth, edgecolor='white', label='Ground truth')
plt.bar(r2, bars2, color='blue', width=barWidth, edgecolor='white', label='Model')

# Add xticks on the middle of the group bars
plt.xlabel('Class', fontsize=fontsize)
plt.ylabel('$p_c$', fontsize=fontsize)
plt.xticks([r + barWidth for r in range(len(bars1))], 
           [str(x) for x in range(1,40)],
           fontsize=fontsize-2)
plt.yticks(fontsize=fontsize)
 
# Create legend & Show graphic
plt.legend()
plt.show()

KL = scipy.stats.entropy(hy,hyhat)

plt.title('KL divergence: {:.6f}'.format(KL), fontsize=fontsize)


#%%

m = df['f1'].mean()
std = 0.05

f1_dummy = np.asarray([x +  np.random.randn()*std for x in np.ones((39))*m])

f1_dummy[f1_dummy> 1] = 1

f1_dummy = f1_dummy + (m-np.mean(f1_dummy))

dm = np.mean(f1_dummy)
np.std(f1_dummy)


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
         df.loc[sort.index, 'f1'], 
         color=[1, 0.3, 0.3],
         marker='+')

ax2.plot(range_,
         f1_dummy,
         color=[0, 1, 0],
         marker='x',
         linestyle='solid')

ax1.set_ylabel('N images', fontsize=fontsize)
ax2.set_ylabel('Metric score', fontsize=fontsize)
ax1.set_xlabel('Class label and size', fontsize=fontsize)


plt.hlines(m,0,max(range_), colors=[1, 0.3, 0.3])

plt.hlines(dm,0,max(range_), colors=[0, 1, 0], linestyle='dotted')

plt.annotate('Macro-averaged F1 score', 
             [25,m],
             xytext=[20, 0.6],
             arrowprops={'arrowstyle':'->'})

plt.legend(['F1a. Macro-avg: {:.2f}'.format(m), 
            'F1b Macro-avg: {:.2f}'.format(dm)], 
           loc='lower right',
           fontsize=fontsize-4)
ax2.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)


for i in range(1,40):
    ax2.axvline(x=i, alpha=0.1)



