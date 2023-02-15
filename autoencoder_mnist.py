import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import timedelta
import sys
sys.path.append('apps\templates')
from apps.templates.docdb_client import DocdbClient

# class MotiDocDBClient(DocdbClient):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def do_work(self):
#         pass

#import server
#singtel = server.DocdbClient()
df = pd.read_csv('anomalydetection/research/sand_box_vhf/data/mnist_test.csv')
anom = df[:1000]
clean = df[1000:]
for i in range(len(anom)):
    # select row from anom
    row = anom.iloc[i]
    # iterate through each element in row
    for i in range(len(row)-1):
        # add noise to element
        row[i+1] = min(255, row[i+1]+random.randint(100,200))
anom['label'] = 1
clean['label'] = 0
an_test = pd.concat([anom, clean])  # join
an_test.sample(frac=1)              # shuffle
an_test.to_csv('anom.csv')          # save
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(48, 50),
            nn.ReLU(),
            nn.Linear(50, 64),
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
            nn.Embedding(10, 16)
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU
        )
    def forward(self, x, y):
        encode = self.enc(x)
        label_encode = self.label_enc(y)
        decode = self.dec(torch.cat([encode, label_encode]))
        return decode
batch_size = 32
lr = 1e-2         # learning rate
w_d = 1e-5        # weight decay
momentum = 0.9
epochs = 15
class Loader(torch.utils.data.Dataset):
    def __init__(self):
        super(Loader, self).__init__()
        self.dataset = ''

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        row = self.dataset.iloc[idx]
        label = row['label']
        row = row.drop(labels={'label'})
        data = torch.from_numpy(np.array(row) / 255).float()
        return data, label
class Train_Loader(Loader):
    def __init__(self):
        super(Train_Loader, self).__init__()
        self.dataset = pd.read_csv(
            'anomalydetection/research/sand_box_vhf/data/mnist_train.csv',
            index_col=False
        )

train_set = Train_Loader()
train_ = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
            drop_last=True     )
metrics = defaultdict(list)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AE()
model.to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

model.train()
start = time.time()
for epoch in range(epochs):
    ep_start = time.time()
    running_loss = 0.0
    for bx, (data) in enumerate(train_):
        sample = model(data.to(device))
        loss = criterion(data.to(device), sample)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss/len(train_set)
    metrics['train_loss'].append(epoch_loss)
    ep_end = time.time()
    print('-----------------------------------------------')
    print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch+1,epochs,epoch_loss))
    print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
end = time.time()
print('-----------------------------------------------')
print('[System Complete: {}]'.format(timedelta(seconds=end-start)))

#
# # Load the MNIST dataset
# mnist_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
#
# # Preprocess the MNIST data
# mnist_data = mnist_data.data.reshape(-1, 784)
#
# # Define the autoencoder architecture
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(784, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 784),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
# # Train the autoencoder
# autoencoder = Autoencoder()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
#
# for epoch in range(100):
#     for i, (img, _) in enumerate(mnist_data):
#         optimizer.zero_grad()
#         output = autoencoder(img)
#         loss = criterion(output, img)
#         loss.backward()
#         optimizer.step()
#
# # Use the autoencoder for anomaly detection
# def detect_anomalies(autoencoder, mnist_data):
#     anomalies = []
#     for i, (img, _) in enumerate(mnist_data):
#         reconstructed_img = autoencoder(img)
#         diff = (reconstructed_img - img).norm().item()
#         if diff > threshold:
#             anomalies.append((i, diff))
#     return anomalies
#
# # Evaluate the results
# anomalies = detect_anomalies(autoencoder, mnist_data)
