#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import inv

np.random.seed(42)
torch.manual_seed(42)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"     
os.environ["CUDA_VISIBLE_DEVICES"]="0" 


# In[2]:

import sys


N = 10000
BATCH_SIZE = 100
dimension = int(sys.argv[1])
activation = sys.argv[2]
number_of_neurons = int(sys.argv[3])

print(dimension) 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(dimension, number_of_neurons)
        self.fc2 = nn.Linear(number_of_neurons, 1)

    def forward(self, x):
        if(activation == 'cos'):
            x = torch.cos(self.fc1(x))
        if(activation == 'exp'):
            x = torch.exp(-self.fc1(x)*self.fc1(x)/2)
        if(activation == 'relu'):
            x = F.relu(self.fc1(x))
        if(activation == 'tanh'):
            x = torch.tanh(self.fc1(x))
        if(activation == 'sigmoid'):
            x = torch.sigmoid(self.fc1(x))
        return self.fc2(x)
    def init_weights(self, features_vectors, bias, weight):
        self.fc1.weight.data = features_vectors
        self.fc1.bias.data = bias
        self.fc2.weight.data = weight
        self.fc2.bias.data.fill_(0.0)
        

# In[3]:

def load_dataset(dimension, l):
    x = np.loadtxt("dim{}_k{}_x.txt".format(dimension, l))
    fraction_ax = np.loadtxt("dim{}_k{}_y.txt".format(dimension, l))
    dataset = [x[:N], fraction_ax[:N]]
    test = [x[N:20000], fraction_ax[N:20000]]
    return dataset, test

# In[4]:


from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

device = torch.device("cuda")

def train_model(N, A, features_vectors, bias, weight, max_epochs=1000):

    # take num_a random values of a from 1 to A - 1
    train_loss = [0 for _ in range(max_epochs)]
    test_loss = [0 for _ in range(max_epochs)]
    

    dataset, test = load_dataset(dimension, A)
    train_data = TensorDataset(torch.Tensor(dataset[0]), torch.Tensor(dataset[1]))
    test_data = TensorDataset(torch.Tensor(test[0]), torch.Tensor(test[1]))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=(20000-N), shuffle=True)
    net = Net().to(device)
    one  = torch.from_numpy(features_vectors).float().to(device)
    two = torch.from_numpy(bias).float().to(device)
    three = torch.from_numpy(weight).float().to(device)
    net.init_weights(one, two, three)
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
        train_loss[epoch] = train_loss[epoch]*BATCH_SIZE/N
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = net(x.view(-1, dimension))
            loss = criterion(output, y.view(-1, 1))
            test_loss[epoch] += loss.item()
        net.eval()

    train_loss = [loss  for loss in train_loss]
    test_loss = [loss for loss in test_loss]

    return test_loss


# In[ ]:


if dimension == 3:
    A_range = np.array([1,3,5,10,20,40,60])
if dimension == 4:
    A_range = np.array([1,3,5,7,10,20,29])
if dimension == 5:
    A_range = np.array([1,3,5,7,10,12,16])
if dimension == 6:
    A_range = np.array([2,3,4,5,7,8,10])

features_vectors_i = []
biases_i = []
weights_i = []
for i, A in enumerate(A_range):
    dataset, test = load_dataset(dimension, A)
    X_train = dataset[0]
    Y_train = np.reshape(dataset[1], [-1,1])
    X_test = test[0]
    Y_test = np.reshape(test[1], [-1,1])

    feature_vectors = np.random.normal(0, 1, size=(dimension,number_of_neurons))
    bias = np.random.normal(0, 1, size=(1,number_of_neurons))
    features_vectors_i.append(np.transpose(feature_vectors))
    biases_i.append(np.reshape(bias, [-1]))
    if activation == 'relu':
        func = lambda x: x * (x > 0)
    if activation == 'exp':
        func = lambda x: np.exp(-0.5*np.multiply(x,x))
    if activation == 'cos':
        func = lambda x: np.cos(x)
    if activation == 'tanh':
        func = lambda x: np.tanh(x)
    if activation == 'sigmoid':
        func = lambda x: 1/(1+np.exp(-x))
    features = func(np.matmul(X_train,feature_vectors)+bias)
    A = np.matmul(features.T, Y_train)
    K = np.matmul(features.T, features)+np.identity(number_of_neurons)
    weights = np.matmul(inv(K), A)
    weights_i.append(np.transpose(weights))

test_losses = []
for i, A in enumerate(A_range):
    test_loss = train_model(N, A, features_vectors_i[i], biases_i[i], weights_i[i])
    test_losses.append(test_loss)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Plotting training and testing losses
plt.figure(figsize=(8, 5))
colors = cm.plasma(np.linspace(0, 1, len(A_range)))
# plot one horizontal line as 1/12 dashed line
#plt.axhline(y=1/12, color='r', linestyle='--', label='baseline 1/12')    
for i, A in enumerate(A_range):
    cc = np.convolve(test_losses[i], np.ones(5)/5, mode='valid')
    plt.plot(cc, linewidth=1, color=colors[i], label='k = {}'.format(round(A,2)))
    np.savetxt('output/activation{}_dim{}_k{}.txt'.format(activation, dimension, A), test_losses[i])
plt.xlabel('Epoch')
plt.ylabel('MSE')
# save plot
plt.ylim(0, 1.2)
plt.legend()
plt.savefig('figs/test_loss_activation{}_dim{}.png'.format(activation, dimension), dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:




