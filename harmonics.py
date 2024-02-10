#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"     
os.environ["CUDA_VISIBLE_DEVICES"]="1" 


# In[2]:

import sys


N = 10000
BATCH_SIZE = 100
dimension = int(sys.argv[1])
activation = sys.argv[2]

print(dimension) 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(dimension, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        if(activation == 'cos'):
            x = torch.cos(self.fc1(x))
        if(activation == 'exp'):
            x = torch.exp(-self.fc1(x)*self.fc1(x)/2)
        if(activation == 'relu'):
            x = F.relu(self.fc1(x))
        return self.fc2(x)


# In[3]:

def load_dataset(dimension, l):
    x = np.loadtxt("dim{}_k{}_x.txt".format(dimension, l))
    fraction_ax = np.loadtxt("dim{}_k{}_y.txt".format(dimension, l))
    dataset = [x, fraction_ax]
    return dataset

# In[4]:


from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

device = torch.device("cuda")

def train_model(N, A, max_epochs=2000, num_a=5):

    # take num_a random values of a from 1 to A - 1
    train_loss = [0 for _ in range(max_epochs)]

    for a in range(num_a):
        dataset = load_dataset(dimension, A)
        train_data = TensorDataset(torch.Tensor(dataset[0]), torch.Tensor(dataset[1]))
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        net = Net().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(max_epochs):
            net.train()
            for batch in train_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = net(x.view(-1, dimension))
                loss = criterion(output, y.view(-1, 1))
                loss.backward()
                optimizer.step()
            train_loss[epoch] += loss.item()
            net.eval()

    train_loss = [loss / num_a for loss in train_loss]

    return train_loss


# In[ ]:

MIN_A = 1
if dimension == 3:
    MAX_A = 22
if dimension == 4:
    MAX_A = 13
if dimension == 5:
    MAX_A = 9
if dimension == 6:
    MAX_A = 7

if dimension == 3:
    A_range = np.arange(MIN_A, MAX_A+1, 4)
if dimension == 4:
    A_range = np.arange(MIN_A, MAX_A+1, 2)
if dimension == 5:
    A_range = np.arange(MIN_A, MAX_A+1, 2)
if dimension == 6:
    A_range = np.arange(MIN_A, MAX_A+1, 2)

#np.logspace(np.log10(MIN_A), np.log10(MAX_A), 10).astype(int)
train_losses = []

for A in tqdm(A_range):
    train_loss = train_model(N, A)
    train_losses.append(train_loss)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Plotting training and testing losses
plt.figure(figsize=(8, 5))
colors = cm.plasma(np.linspace(0, 1, len(A_range)))
# plot one horizontal line as 1/12 dashed line
#plt.axhline(y=1/12, color='r', linestyle='--', label='baseline 1/12')    
for i, A in enumerate(A_range):
    plt.plot(train_losses[i], linewidth=1, color=colors[i], label='k = {}'.format(round(A,2)))
    np.savetxt('output/activation{}_dim{}_k{}.txt'.format(activation, dimension, A), train_losses[i])
plt.xlabel('Epoch')
plt.ylabel('MSE')
# save plot
plt.ylim(0, 0.5)
plt.legend()
plt.savefig('figs/train_loss_activation{}_dim{}.png'.format(activation, dimension), dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:




