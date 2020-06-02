# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:30:42 2020

@author: Mikko Impiö
"""
import random
import os
import pandas as pd

###### DATAFRAME FUNCTIONS ###############

def create_df(csvpath, img_path, partial_dataset=False,seed=123):
    df = pd.read_csv(csvpath,
                     delimiter=' ',
                     header=None)
    
    # take only the first 10% of datasets for testing
    random.seed(seed)
    if partial_dataset:
        
        percent = 0.05
        
        df = df.loc[random.sample(range(0,len(df)), int(percent*len(df))),:]
    # end if
        
    # clean up the splits
    df = df.iloc[:,[0,1]]
    df.columns = ["path","label"]
    df.loc[:,"path"] = df.loc[:,"path"].apply(lambda x: x.replace("\\",os.sep))
    df['path'] = df['path'].map(lambda x: img_path+x)
    return df
