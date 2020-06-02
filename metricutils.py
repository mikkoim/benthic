# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:02:25 2020

@author: Mikko Impiö
"""

from sklearn import metrics
import numpy as np
import scipy
import pandas as pd

def calc_basic_metrics(y, yhat):
    acc = metrics.accuracy_score(y, yhat)
    
    av = 'macro'
    
    precision = metrics.precision_score(y, yhat, average=av, zero_division=0)
    
    recall = metrics.recall_score(y, yhat, average=av, zero_division=0)
    
    f1 = metrics.f1_score(y, yhat, average=av, zero_division=0)
    
    print('Accuracy: {:.4f}'.format(acc))
    
    print('Precision: {:.4f}'.format(precision) )
    
    print('Recall: {:.4f}'.format(recall) )
    
    print('F1 score: {:.4f}'.format(f1) )
    
    print('################################')
    
    return acc, precision, recall, f1


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

def bin_classification(y, yhat, range_):
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
    r['gmean'] = []
    r['gmeasure'] = []
        
    for l in range(range_[0],range_[1]):
        
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
        
        f1 = (2*tpr*ppv)/(tpr+ppv)
        acc= (tp+tn)/(p+n)
        mcc = calc_mcc(tp, fp, tn, fn)
        bacc = (tpr+tnr)/2
        bayes = (p_ba*p_a)/p_b
        gmean = np.sqrt(tpr*tnr)
        gmeasure = np.sqrt(tpr*ppv)
        
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
        r['gmean'].append(gmean)
        r['gmeasure'].append(gmeasure)
        
        
    df = pd.DataFrame.from_dict(r)
    df = df.set_index(np.arange(range_[0],range_[1]))
        
    return df