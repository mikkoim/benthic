# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:29:37 2020

@author: Mikko Impiö
"""

import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import torch

from loadbmtorch import create_df

#Load the ready-made splits


if platform.system() == 'Linux':
    datapath = '/home/mikko/Documents/kandi/data/IDA/Separate lists with numbering/Machine learning splits'
    img_path = '/home/mikko/Documents/kandi/data/IDA/Images/'
else:
    datapath = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Separate lists with numbering\\Machine learning splits'
    img_path = 'C:\\koodia\\kandi\\FIN Benthic2\\IDA\\Images\\'

split = 1

train_fname = 'train'+str(split)+'.txt'
test_fname = 'test'+str(split)+'.txt'
val_fname = 'val'+str(split)+'.txt'

part_dat = True

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

#%% Create torch dataloader


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.preprocessing import LabelBinarizer


class BenthicDataset(Dataset):
    def __init__(self, df, classes, transform=None):
        self.df = df
        self.transform = transform
        self.lb = LabelBinarizer().fit(np.arange(1,classes+1))
        
        self.labels = self.lb.transform(df.label)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        fpath = self.df.iloc[idx,0]
        
        img = Image.open(fpath)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
            
        sample = (img,label)

        return sample
    
makeds = lambda df: BenthicDataset(
                    df,
                    classes=39,
                    transform=transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()]))

train_ds = makeds(df_train)
val_ds = makeds(df_val)

makedl = lambda ds: DataLoader(ds, 
                               batch_size=8,
                               shuffle=True,
                               num_workers=4)

train_dl = makedl(train_ds)
val_dl = makedl(val_ds)

device = torch.device("cuda:0")

#%% Training torch model
from torchvision import models
import torch.nn as nn
import torch.optim as optim

model = models.alexnet(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
n_inputs = model.classifier[6].in_features
n_classes = 39

model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, n_classes),                   
                      nn.LogSoftmax(dim=1))

model = model.to('cuda')

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

from trainbmtorch import train_model

model_ft = train_model(model, criterion, optimizer, scheduler,
                       num_epochs=25)