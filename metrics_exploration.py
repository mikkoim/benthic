# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:16:02 2020

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


preds = np.load(os.path.join('preds','18-01-2020_cont_colab.npy'))
yhat = np.argmax(preds,axis=1)+1
y = df_test['label'].to_numpy()


#%% Create taxonomic hierarchy table


label_df = pd.read_csv(os.path.join(datapath, 'train1.txt'),
                       delimiter=' ',
                       header=None)

label_df.columns = ['path', '5-taxa', '4-species', '3-genus', '2-family', '1-order']
label_df = label_df.iloc[:,1:].drop_duplicates()
label_df = label_df.sort_values('5-taxa')
label_df.index = range(1,40)

def get_H_level(row):
    return 4 - sum(row[1:] == 0)

label_df['H'] = label_df.apply(get_H_level,axis=1)

#%% LCSE loss function

def deepest_common_ancestor(y, yhat):
    
    if y == yhat:
        return 0
    
    row = label_df.loc[y,:]
    rowhat = label_df.loc[yhat,:]
    
    m = row == rowhat
    match = m[:-1]
    
    h = 4-sum(match)
    
    return h

h = np.zeros((39,39))
    
for i in range(39):
    for j in range(39):
        H = np.max([label_df.loc[i+1,'H'],
                   label_df.loc[j+1,'H']])
        h[i,j] = deepest_common_ancestor(i+1,j+1)/H
        
def LCSE(y, yhat):
    assert len(y) == len(yhat)
    
    L = np.zeros((len(y),1))
    
    for i in range(len(y)):
        score = h[y[i]-1, yhat[i]-1]
        L[i] = score   
        
    return np.sum(L)/len(y)
    
       
#%% Accuracies
        
lcse = LCSE(y, yhat)
ce = np.sum(y != yhat)/len(y)

print("LCSE: {:.3f}".format(lcse))
print("CE: {:.3f}".format(ce))

#%% Insect aggretation

from combine_insects import add_insect_class, add_yhat

df_test_preds = add_insect_class(df_test)

# adding predictions to dataframe for insect-wise prediction
df_test_preds = add_yhat(df_test_preds,yhat)

dfg = df_test_preds.groupby(['label','insect'],as_index=False)['pred'].agg(lambda x:x.value_counts().index[0])

y_ins = dfg['label'].to_numpy()
yhat_ins = dfg['pred'].to_numpy()

#%%
lcse_ins = LCSE(y_ins, yhat_ins)
ce_ins = np.sum(y_ins != yhat_ins)/len(y_ins)

print("insect LCSE vote: {:.3f}".format(lcse_ins))
print("insect CE vote: {:.3f}".format(ce_ins))


#%% Class histogram exploration

dr = 0.015 # diff range

plt.figure(0)

plt.subplot(3,3,1)
hy = np.histogram(y,bins=39, normed=True)[0]
plt.bar(np.arange(1,40), hy)
plt.title('Image-wise')
plt.ylabel('True labels')

plt.subplot(3,3,2)
hy_ins = np.histogram(y_ins,bins=39, normed=True)[0]
plt.bar(np.arange(1,40), hy_ins)
plt.title('Insect-wise')

plt.subplot(3,3,4)
hyhat = np.histogram(yhat,bins=39, normed=True)[0]
plt.bar(np.arange(1,40), hyhat, color='red')
plt.ylabel('Predicted labels')

plt.subplot(3,3,5)
hyhat_ins = np.histogram(yhat_ins,bins=39, normed=True)[0]
plt.bar(np.arange(1,40), hyhat_ins, color='red')

#Deltas
plt.subplot(3,3,3)
delta = hy-hy_ins
plt.bar(np.arange(1,40),delta, color='green')
plt.axis([0,40, -dr,dr])
plt.title('Image-Insect diff')

plt.subplot(3,3,6)
deltahat = hyhat-hyhat_ins
plt.bar(np.arange(1,40),deltahat, color='green')
plt.axis([0,40, -dr,dr])

plt.subplot(3,3,7)
delta_ins = hy-hyhat
plt.bar(np.arange(1,40),delta_ins, color='green')
plt.axis([0,40, -dr,dr])
plt.ylabel('True-pred diff')

plt.subplot(3,3,8)
deltahat_ins = hy_ins-hyhat_ins
plt.bar(np.arange(1,40),deltahat_ins, color='green')
plt.axis([0,40, -dr,dr])
#%% Basic metrics

from sklearn import metrics

acc = metrics.accuracy_score(y, yhat)
acc_ins = metrics.accuracy_score(y_ins, yhat_ins)

av = 'macro'

precision = metrics.precision_score(y, yhat, average=av, zero_division=0)
precision_ins = metrics.precision_score(y_ins, yhat_ins, average=av, zero_division=0)

recall = metrics.recall_score(y, yhat, average=av, zero_division=0)
recall_ins = metrics.recall_score(y_ins, yhat_ins, average=av, zero_division=0)

f1 = metrics.f1_score(y, yhat, average=av, zero_division=0)
f1_ins = metrics.f1_score(y_ins, yhat_ins, average=av, zero_division=0)

print('Accuracy: {:.4f}'.format(acc))
print('Accuracy insect: {:.4f}\n'.format(acc_ins) )

print('Precision: {:.4f}'.format(precision) )
print('Precision insect: {:.4f}\n'.format(precision_ins) )

print('Recall: {:.4f}'.format(recall) )
print('Recall insect: {:.4f}\n'.format(recall_ins) )

print('F1 score: {:.4f}'.format(f1) )
print('F1 score insect: {:.4f}\n'.format(f1_ins) )

#%% Precision and recall

def confusion_matrix(y, yhat, l):
    
    ybin = np.zeros(y.shape)
    ybin[y==l] = 1
    
    yhatbin = np.zeros(yhat.shape)
    yhatbin[yhat==l] = 1
    
    S = np.int32(ybin+yhatbin)
    
    tp = np.sum(S==2)
    fp = np.sum(scipy.logical_and((S==1),yhatbin))
    tn = np.sum(scipy.logical_and((S==0),(yhatbin==0)))
    fn = np.sum(scipy.logical_and((S==1),ybin))
    
    p = np.sum(ybin==1)
    n = np.sum(ybin==0)
        
    assert tp+fn == p
    assert tn+fp == n
    assert p+n == len(y)
    
    return tp, fp, tn, fn

def calc_mcc(tp, fp, tn, fn):
    
    N = tn+tp+fn+fp
    S = (tp+fn)/N
    P = (tp+fp)/N
    
    return (tp/N - S*P)/np.sqrt(P*S*(1-S)*(1-P))

def bin_classification(y, yhat):
    assert len(y) == len(yhat), "Arrays not same length"
    r = {}
    
    r['tp'] = []
    r['fp'] = []
    r['tn'] = []
    r['fn'] = []
    
    r['p'] = []
    r['n'] = []
    
    r['tpr'] = []
    r['tnr'] = []
    
    r['ppv'] = []
    r['npv'] = []
    
    r['f1'] = []
    r['mcc'] = []
    r['acc'] = []
    r['bacc'] = []
    r['bayes'] = []
        
    for l in range(1,40):
        
        tp, fp, tn, fn = confusion_matrix(y, yhat, l)
    
        p = tp+fn
        n = fp+tn
    
        tpr = tp/p
        tnr = tn/n
        
        ppv = tp/(tp+fp)
        npv = tn/(tn+fn)
                
        #bayes
        p_a = p/(p+n)
        p_ba = tpr
        p_bnota = (1-tnr)
        p_b = (p_ba*p_a) + (p_bnota*(1-p_a))
        
        
        f1 = (2*tp)/(2*tp+fp+fn)        
        acc= (tp+tn)/(p+n)
        mcc = calc_mcc(tp, fp, tn, fn)
        bacc = (tpr+tnr)/2
        bayes = (p_ba*p_a)/p_b
        
        r['tp'].append(tp)
        r['fp'].append(fp)
        r['tn'].append(tn)
        r['fn'].append(fn)
        
        r['p'].append(p)
        r['n'].append(n)
        
        r['tpr'].append(tpr)
        r['tnr'].append(tnr)
        
        r['ppv'].append(ppv)
        r['npv'].append(npv)
        
        r['f1'].append(f1)
        r['mcc'].append(mcc)
        r['acc'].append(acc)
        r['bacc'].append(bacc)
        r['bayes'].append(bayes)
        
        
    df = pd.DataFrame.from_dict(r)
    df = df.set_index(np.arange(1,40))
        
    return df
        

df = bin_classification(y, yhat)
df_ins = bin_classification(y_ins, yhat_ins)

df = df.fillna(0)
df_ins = df_ins.fillna(0)

#%% 

def plot_asc_bar(S):
    plt.title(S.name)
    sort = S.sort_values()
    plt.bar(range(1,40), sort.values)
    plt.axis([1,40,sort.min(),sort.max()])
    _ = plt.xticks(np.arange(1,40), sort.index)
    
plt.subplot(2,2,1)
plot_asc_bar(df['tpr'])
plt.subplot(2,2,2)
plot_asc_bar(df['ppv'])
plt.subplot(2,2,3)
plot_asc_bar(df['tnr'])
plt.subplot(2,2,4)
plot_asc_bar(df['npv'])

#%%
sort = df['p'].sort_values()
fig, ax1 = plt.subplots()
ax1.set_title('Accuracy vs class size')
ax1.bar(range(1,40), sort.values)
ax2 = ax1.twinx()  
ax2.plot(range(1,40), df.loc[sort.index, 'acc'], color='y')

sort = df['p'].sort_values()
fig, ax1 = plt.subplots()
ax1.set_title('F1 vs class size')
ax1.bar(range(1,40), sort.values)
ax2 = ax1.twinx()  
ax2.plot(range(1,40), df.loc[sort.index, 'f1'], color='b')

sort = df['p'].sort_values()
fig, ax1 = plt.subplots()
ax1.set_title('MCC vs class size')
ax1.bar(range(1,40), sort.values)
ax2 = ax1.twinx()  
ax2.plot(range(1,40), df.loc[sort.index, 'mcc'], color='r')

sort = df['p'].sort_values()
fig, ax1 = plt.subplots()
ax1.set_title('BAcc vs class size')
ax1.bar(range(1,40), sort.values)
ax2 = ax1.twinx()  
ax2.plot(range(1,40), df.loc[sort.index, 'bacc'], color='g')

#%%
sort = df['p'].sort_values()
fig, ax1 = plt.subplots()
ax1.set_title('Class size vs metrics')
ax1.bar(range(1,40), sort.values)
ax2 = ax1.twinx()  
ax2.plot(range(1,40), df.loc[sort.index, 'f1'], color='b')
ax2.plot(range(1,40), df.loc[sort.index, 'mcc'], color='r', linestyle='dashed')
ax2.plot(range(1,40), df.loc[sort.index, 'bacc'], color='g')
ax2.plot(range(1,40), df.loc[sort.index, 'gmean'], color='y')

plt.legend(['f1','mcc','bacc','gmean'])

#%%
sort = df['f1'].sort_values()
fig, ax1 = plt.subplots()
ax1.set_title('Class size vs metrics, F1 ascending')
ax1.bar(range(1,40), df.loc[sort.index, 'p'], color='b')
ax2 = ax1.twinx()  
ax2.plot(range(1,40), sort.values, color='b')
ax2.plot(range(1,40), df.loc[sort.index, 'mcc'], color='r', linestyle='dashed')
ax2.plot(range(1,40), df.loc[sort.index, 'bacc'], color='g')

plt.legend(['f1','mcc','bacc'])

#%% Cumulative scores vs size
sort = df['p'].sort_values()
df_sort = df.loc[sort.index]

cum_f1 = []
for i in range(len(sort)):
    
    cumsum = lambda col, i: df_sort[col].iloc[:i].sum()
    
    tp = cumsum('tp',i)
    fp = cumsum('fp',i)
    fn = cumsum('fn',i)
    
    f1 = (2*tp)/(2*tp+fp+fn) 
    cum_f1.append(f1)
    
cum_f1 = np.nan_to_num(np.array(cum_f1))


fig, ax1 = plt.subplots()
ax1.set_title('Cumulative f1 vs class size')
ax1.bar(range(1,40), df.loc[sort.index, 'p'], color='b')
ax2 = ax1.twinx()  
ax2.plot(range(1,40), cum_f1, color='r')


#%%
def low_performers(df, n):
    tnr = df['tnr'].sort_values()[0:n]
    tpr = df['tpr'].sort_values()[0:n]
    ppv = df['ppv'].sort_values()[0:n]
    npv = df['npv'].sort_values()[0:n]
    
    tnr_low = set(tnr.index)
    tpr_low = set(tpr.index)
    ppv_low = set(ppv.index)
    npv_low = set(npv.index)
    
    union_ = tnr_low.union(tpr_low).union(ppv_low).union(npv_low)  
    pos = tpr_low.intersection(ppv_low)
    neg = tnr_low.intersection(npv_low)
    
    
    d = {}
    for v in union_:
        d[v] = 0
        for s in [tnr_low, tpr_low, ppv_low, npv_low]:
            if v in s:
                d[v] += 1
                
    df = pd.DataFrame.from_dict(d, orient='index')
    df.columns = ['count']
    return df, pos, neg
    
low, pos, neg = low_performers(df, 5)

#%% Number of images and F1 score  

def pair_scatter(x, y, xlabel, ylabel, ax=None):
    if not ax:
        fig, ax = plt.subplots()
        
    ax.scatter(x.values, y.values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for i, c in enumerate(df.index):
        ax.annotate(c, (x.values[i], y.values[i]))
    
pair_scatter(df['p'], df['f1'], 'N images', 'F1 scores')

fig, axs = plt.subplots(2,2)
pair_scatter(df['p'], df['tpr'], 'N images', 'TPR',ax=axs[0][0])
pair_scatter(df['p'], df['ppv'], 'N images', 'PPV',ax=axs[0][1])
pair_scatter(df['p'], df['tnr'], 'N images', 'TNR',ax=axs[1][0])
pair_scatter(df['p'], df['npv'], 'N images', 'NPV',ax=axs[1][1])

#%%

pair_scatter(df['tpr'], df['f1'], 'Recall', 'F1 scores')

fig, ax = plt.subplots()
pair_scatter(df['ppv'], df['f1'], 'Precision', 'F1 scores', ax=ax)
pair_scatter(df['ppv'], df['tpr'], 'precision', 'recall', ax=ax)

#%% Dominance

dom = df['tpr']- df['tnr']

sort_col = 'p'

sort = df[sort_col].sort_values()

dom = dom.loc[sort.index]
fig, ax1 = plt.subplots()
ax1.bar(range(1,40), df.loc[sort.index, 'p'], color='b')

ax2 = ax1.twinx() 
for i, v in enumerate(sort):
    ax1.text(i + .75, v + sort.max()/100, str(sort.index[i]), color='blue', 
                                     fontsize='10') 
    
ax2.plot(range(1,40), dom, color='r')


#%% Bayes
def calc_bayes(df):
    P_A = df['p']/(df['p']+df['n'])
    P_B = df['tpr']*P_A + (1-df['tpr'])*(1-P_A)
    
    P_BA = df['tpr']
    
    return (P_BA*P_A)/P_B

bp = calc_bayes(df)

sort_col = 'p'

sort = df[sort_col].sort_values()

bp = bp.loc[sort.index]
fig, ax1 = plt.subplots()
ax1.bar(range(1,40), df.loc[sort.index, 'p'], color='b')

ax2 = ax1.twinx() 
for i, v in enumerate(sort):
    ax1.text(i + .75, v + sort.max()/100, str(sort.index[i]), color='blue', 
                                     fontsize='10') 
    
ax2.plot(range(1,40), bp, color='r')

#%% Write report

r = metrics.classification_report(y, yhat)
with open('18-01-2020_cont_colab_split2_report.txt','w+') as f:
    f.write(r)
    
r_ins = metrics.classification_report(y_ins, yhat_ins)
with open('18-01-2020_cont_colab_split2_report_ins.txt','w+') as f:
    f.write(r_ins)

#%% report DF

C = metrics.confusion_matrix(y, yhat, normalize='true')
C_ins = metrics.confusion_matrix(y_ins, yhat_ins, normalize='true')


def make_reportdf(y, yhat):
    d = metrics.classification_report(y, yhat, output_dict=True, zero_division=0) 
    return pd.DataFrame.from_dict(d).transpose()

report = make_reportdf(y,yhat)
report_ins = make_reportdf(y_ins, yhat_ins)

#%% Bayes stuff with Conf. matrix

C = metrics.confusion_matrix(y, yhat, normalize='false')

preds = C.sum(axis=0)
trues = C.sum(axis=1)

p_a = trues/np.sum(trues)
p_b = preds/np.sum(preds)

p_ba = np.diag(C)/trues
p_ba = np.nan_to_num(p_ba)

bayes = (p_ba*p_a)/p_b

import scipy

harm_bayes = scipy.stats.hmean(bayes+1e-7)
mean_bayes = bayes.mean() 


#%% confusion bar plots

sort = df['p'].sort_values()

tpr = df['tpr'].values
fpr = 1-df['tnr'].values
fnr = 1-tpr

tpr = tpr[sort.index-1]
fpr = fpr[sort.index-1]
fnr = fnr[sort.index-1]

ind = np.arange(39)

p1 = plt.bar(ind, tpr, color='green')
p2 = plt.bar(ind, fpr,bottom=tpr, color='orange')
p3 = plt.bar(ind, -fnr, color='red')


#%%
import seaborn as sns
            cmap='coolwarm',
            annot=True, 

sns.heatmap(C, 
            fmt='.2f', 
            annot_kws={'size':6},
            cbar=False,
            xticklabels=range(1,40),
            yticklabels=range(1,40),
            square=True)


cmdf = df.iloc[:,6:10]
sns.heatmap(cmdf.corr(),annot=True, cmap='hot')
sns.pairplot(cmdf)

#%%
def classvar(C):
    s = []
    for row in C:
        vals = row[abs(row)>1e-24]
        sigma = np.std(vals)
        s.append(sigma)
        
    return np.array(s)
    
s = classvar(C)

plt.scatter(s, df['ppv'])


#%%
r =  [np.arange(39)]
barwidth = 1/11
h = np.zeros((10,39))
for i in range(10):
    r.append([x + barwidth for x in r[i]]) 
    
    y = splits['train'][i]
    h[i,:] = np.histogram(y,bins=39)[0]/len(y)
    
    plt.bar(r[i+1], h[i,:], color='blue', width=barwidth, zorder=1)
    
plt.scatter(np.arange(39)+.5, href, marker='_',color='red', s=100, zorder=2)


#%% Score histograms

c = np.concatenate([df['tpr'].values, df['ppv'].values])

plt.hist(c, bins=10)

#%%

def labels_to_class_weights(labels, nc=80): 
    # Get class weights (inverse frequency) from training labels 
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO 
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh] 
    weights = np.bincount(classes, minlength=nc)  # occurences per class 
    weights[weights == 0] = 1  # replace empty bins with 1 
    weights = 1 / weights  # number of targets per class 
    weights /= weights.sum()  # normalize 
    return torch.Tensor(weights) 

#%%


labels = np.histogram(df_train['label'], bins=39)[0]
weights = labels
weights = 1-(labels/labels.max())
weights[np.argmin(weights)] = np.sort(weights)[1]

plt.bar(range(1,40),weights)


#%%

alpha = np.histogram(df_train['label'], bins=39)[0]
alpha[alpha==0] = np.sort(alpha)[1]
alpha = 1 / alpha  
alpha /= alpha.sum() 

plt.bar(range(1,40),alpha)


#%% Precision-recall

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer().fit(range(1,40))
Y_test = lb.transform(y)
y_score = preds
n_classes = 39

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


#%%
from itertools import cycle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = cm.get_cmap('viridis', n_classes)
fontsize= 18


plt.figure(figsize=(10, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('F1={0:0.1f}'.format(f_score), xy=(f_score**1.8+0.02, 1.01))

lines.append(l)
labels.append('iso-F1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('Micro-average, AUC = {0:0.2f}'
              ''.format(average_precision["micro"]))

for i in range(n_classes):
    l, = plt.plot(recall[i], precision[i], color=colors(i/n_classes), lw=2)
    lines.append(l)
    labels.append('{0}, AUC = {1:0.2f}'
                  ''.format(i+1, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25, right=0.7)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall',fontsize=fontsize)
plt.ylabel('Precision',fontsize=fontsize)
plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5), prop=dict(size=7))

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

y_test = Y_test
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#%%
lw = 2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()