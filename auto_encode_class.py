import torch
import pandas as pd
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def shaffel_label(anom):
    for i2 in range(len(anom)):
        for j in range(20):
            new_l = random.randint(0,9)
            if new_l != anom.label.iloc[i2]:
               anom.label.iloc[i2]=  new_l
               break
    return anom        
def add_noise(anom):
    for i2 in range(len(anom)):
        # select row from anom
        # iterate through each element in row
       # plt.imshow(np.reshape(anom.iloc[i2][1:].values,(28,28)));plt.show()
        for i in range(len(anom.iloc[i2])-1):
            # add noise to element
            anom.iloc[i2][i+1] = min(255, anom.iloc[i2][i+1]+random.randint(70,100)-5)
       # plt.imshow(np.reshape(anom.iloc[i2][1:].values,(28,28)));plt.show()    
    return anom  

def los_dist(anom,model,criterion, device ):
    loss_dist = []
    for i in range(len(anom)):
        data = torch.from_numpy(np.array(anom.iloc[i][1:-1])/255).float()
        sample = model(data.to(device), torch.from_numpy(np.array(anom.iloc[i]['label'])).to(device))
        loss = criterion(data.to(device), sample)
        loss_dist.append(loss.item())
    return loss_dist    

def conf_matrix(df, loss_dist, upper_threshold):
    ddf = pd.DataFrame(columns=df.columns)
    tp = 0;fp = 0;tn = 0;fn = 0;total_anom = 0
    for i in range(len(loss_dist)):
        total_anom += df.iloc[i]['anomaly_label']
        if loss_dist[i] >= upper_threshold:
            n_df = pd.DataFrame([df.iloc[i]])
            n_df['loss'] = loss_dist[i]
            ddf = pd.concat([df,n_df], sort = True)
            if float(df.iloc[i]['anomaly_label']) == 1.0:
                tp += 1
            else:
                fp += 1
        else:
            if float(df.iloc[i]['anomaly_label']) == 1.0:
                fn += 1
            else:
                tn += 1
    return tp,fp,fn,tn, total_anom

class Loader(torch.utils.data.Dataset):
    def __init__(self, with_time,**kwargs):
        self.with_time = with_time
        super(Loader, self).__init__(**kwargs)
        self.dataset = ''

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        random_ind = torch.randint(0, self.__len__(), (1,))
        row_out = self.dataset.iloc[np.array(random_ind)[0]]  
        if (idx % 1) == 0: # 1 means i=o
            row_out = self.dataset.iloc[idx] 
        row = self.dataset.iloc[idx]
        label_out = row_out['label'] if self.with_time else 5
        row = row.drop(labels={'label'})
        row_out = row_out.drop(labels={'label'})
        data = torch.from_numpy(1*np.array(row) / 255).float()
        data_out = torch.from_numpy(np.array(row_out) / 255).float()
        return data,label_out, data_out# torch.from_numpy(np.array(label_out))#, data_out
class Train_Loader(Loader):
    def __init__(self,with_time):
        super(Train_Loader, self).__init__(with_time=with_time)
        self.dataset = pd.read_csv(
            'anomalydetection/research/sand_box_vhf/data/mnist_train.csv',
            index_col=False
        )


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        enc_out=32
        time_nn=4
        self.enc = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,64 ),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, enc_out),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(enc_out +time_nn, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU()
        )
        self.label_enc =  nn.Sequential(
            nn.Embedding(10, 16),
            nn.ReLU(),
            nn.Linear(16, time_nn),
            nn.ReLU()
        )
    def forward(self, x, y):
        encode = self.enc(x)
        label_encode = self.label_enc(y)
        decode = self.dec(torch.hstack([encode, label_encode]))
        return decode