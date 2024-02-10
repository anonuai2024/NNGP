#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)


# In[2]:


N=10000
def load_dataset(dimension, l):
    x = np.loadtxt("dim{}_k{}_x.txt".format(dimension, l))
    y = np.loadtxt("dim{}_k{}_y.txt".format(dimension, l))
    dataset = [x[:N], y[:N]]
    test = [x[N:20000], y[N:20000]]
    return dataset, test


# In[3]:


dimension = 5
activation = 'tanh'

MIN_A = 1
if dimension == 3:
    MAX_A = 62
if dimension == 4:
    MAX_A = 29
if dimension == 5:
    MAX_A = 16
if dimension == 6:
    MAX_A = 7
if dimension == 3:
    A_range = np.array([1,3,5,10,20,40,60])
if dimension == 4:
    A_range = np.array([1,3,5,7,10,20,29])
if dimension == 5:
    A_range = np.array([1,3,5,7,10,12,16])
if dimension == 6:
    A_range = np.array([2,3,4,5,7,8,10])

# In[ ]:


from numpy.linalg import inv

neurons_range = np.array([100,200,400,800,3200,12800])
#range(1, 6000, 1000)

test_losses = []
for A in A_range:
    dataset, test = load_dataset(dimension, A)
    X_train = dataset[0]
    Y_train = np.reshape(dataset[1], [-1,1])
    X_test = test[0]
    Y_test = np.reshape(test[1], [-1,1])
    Test_mse = []
    for nNeurons in neurons_range:
        MSEs = []
        for trial in range(1):
            feature_vectors = np.random.normal(0, 1, size=(dimension,nNeurons))
            bias = np.random.normal(0, 1, size=(1,nNeurons))
            if activation == 'relu':
                func = lambda x: x * (x > 0)
            if activation == 'exp':
                func = lambda x: np.exp(-0.5*np.multiply(x,x))
            if activation == 'cos':
                func = lambda x: np.cos(x)
            if activation == 'sigma_1':
                func = lambda x: x * (x > 0) - (x-1) * (x > 1)
            if activation == 'sigmoid':
                func = lambda x: 1 / (1 + np.exp(-x))
            if activation == 'tanh':
                func = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            features = func(np.matmul(X_train,feature_vectors)+bias)
            A = np.matmul(features.T, Y_train)
            K = np.matmul(features.T, features)+np.identity(nNeurons)
            weights = np.matmul(inv(K), A)
            test_features = func(np.matmul(X_test,feature_vectors)+bias)
            Y_pred = np.matmul(test_features, weights)
            MSE = np.mean(np.square(Y_pred-Y_test))
            MSEs.append(MSE)
        Test_mse.append([np.mean(np.array(MSEs))])
    test_losses.append(Test_mse)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import make_interp_spline, BSpline



# Plotting training and testing losses
plt.figure(figsize=(8, 5))
colors = cm.plasma(np.linspace(0, 1, len(A_range)))
# plot one horizontal line as 1/12 dashed line
#plt.axhline(y=1/12, color='r', linestyle='--', label='baseline 1/12')    
plt.legend(title = "Legend Title") 
plt.title(r"$n=5, \sigma(x)=tanh(x)$") 
for i, A in enumerate(A_range):
    xnew = np.log(np.linspace(np.min(neurons_range), np.max(neurons_range), 300))
    spl = make_interp_spline(np.log(neurons_range), test_losses[i], k=3)
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth, linewidth=1.0, color=colors[i], label='k = {}'.format(round(A,2)))
    np.savetxt('output/activation{}_dim{}_k{}.txt'.format(activation, dimension, A), test_losses[i])
plt.xlabel('log(Number of neurons)')
plt.ylabel('MSE')
# save plot
plt.ylim(0, 2.2)
plt.legend()
plt.savefig('figs/test_loss_activation{}_dim{}.png'.format(activation, dimension), dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:




