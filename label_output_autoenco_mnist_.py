import random
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import timedelta
import sys
sys.path.append('anomalydetection/research/sand_box_vhf')
#from apps.templates.docdb_client import DocdbClient
from  auto_encode_class    import Train_Loader,AE, Loader, add_noise, conf_matrix, los_dist, shaffel_label
df = pd.read_csv('anomalydetection/research/sand_box_vhf/data/mnist_test.csv')
df['anomaly_label'] = 1
df = df[:2000]
th = 1000
anom = df[:th]
anom = shaffel_label(anom)
clean = df[th:]
#anom = add_noise(anom)
anom.loc[:,'anomaly_label'] = 1
clean.loc[:,'anomaly_label']  =  0
an_test = pd.concat([anom, clean])  # join
an_test = an_test.sample(frac=1).reset_index(drop=True)              # shuffle
an_test.to_csv('anomalydetection/research/sand_box_vhf/data/anom.csv')          # save
batch_size = 256
lr = 1e-3        # learning rate
w_d = 1e-5        # weight decay
momentum = 0.9
epochs = 20
with_time= False
train_set = Train_Loader(with_time)
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
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_d)

model.train()
start = time.time()
for epoch in range(epochs):
    ep_start = time.time()
    running_loss = 0.0
    for bx, (imgin_label_img_out) in enumerate(train_):
        img_out = (imgin_label_img_out[2])
        imgin_label =  (imgin_label_img_out[0],imgin_label_img_out[1])
        sample_out = model(imgin_label[0].to(device), imgin_label[1].to(device)) # .to(device)
        if epoch>110:
             k=99;plt.imshow(torch.reshape(sample_out[k], (28, 28)).cpu().data.numpy());plt.title(f'model {imgin_label[1][k].to(device)}');plt.show()
             plt.imshow(torch.reshape(img_out[k], (28, 28)).cpu().data.numpy());plt.title(f'img_out{imgin_label[1][k].to(device)}');plt.show()
             plt.imshow(torch.reshape(imgin_label[0][k].to(device), (28, 28)).cpu().data.numpy());plt.title(f' img in {imgin_label[1][k].to(device)}');plt.show()
        loss = criterion(img_out.to(device), sample_out)
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
model.eval()
anom_all = pd.read_csv('anomalydetection/research/sand_box_vhf/data/anom.csv', index_col=[0])
anom = anom_all[:1000]
anom_test = anom_all[1000:]
#for bx, data in enumerate(test_):
loss_dist = los_dist(anom,model,criterion, device)
anom['loss_dist'] = loss_dist
anom.sort_values(by=['loss_dist'],ascending=True,inplace=True)
# find the upper_threshold upon some fp ratio
fp_ratio = 0.04
for los in loss_dist:
    mask = anom['loss_dist'] > los
    fp_rate = len((anom['anomaly_label'].loc[mask]==0)) / len((anom.anomaly_label==0))
    if fp_rate<fp_ratio:
        upper_threshold = los
        break
#df = pd.read_csv('anomalydetection/research/sand_box_vhf/data/anom.csv', index_col=[0])# pd.read_csv('data/anom.csv', index_col=[0])
lower_threshold = 0.0
loss_dist = los_dist(anom_test,model,criterion, device) 
tp,fp,fn,tn, total_anom = conf_matrix(anom_test, loss_dist, upper_threshold)
print(f'with time={with_time}')            
print('[TP] {}\t[FP] {}\t[MISSED] {}'.format(tp, fp, total_anom-tp))
print('[TN] {}\t[FN] {}'.format(tn, fn))
plt.figure(figsize=(12,6))
plt.title('Loss Distribution')
sns.distplot(loss_dist,bins=100,kde=True, color='blue')
plt.axvline(upper_threshold, 0.0, 10, color='r')
plt.axvline(lower_threshold, 0.0, 10, color='b')
plt.show();
dd=1
# class MotiDocDBClient(DocdbClient):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def do_work(self):
#         pass
#import server
#singtel = server.DocdbClient()
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
