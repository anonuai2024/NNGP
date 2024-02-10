#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA

dimension = 3

def kernel_eig_fast (activation):
    N = 20000
    M = 20000
    A = np.zeros((N,N))
    x = np.random.normal(0, 1, size=(N, dimension))
    x = x/np.reshape(np.sqrt(np.sum(np.multiply(x,x), axis = 1)), [-1,1])
    y = np.random.normal(0, 1, size=(dimension, M))
    points_vs_feat = activation(np.matmul(x,y))
    points_vs_feat = np.matmul(points_vs_feat,points_vs_feat.T)/M
    return LA.eigh(points_vs_feat)

def relu_analytical():
    N = 20000
    A = np.zeros((N,N))
    x = np.random.normal(0, 1, size=(N, dimension))
    x = x/np.reshape(np.sqrt(np.sum(np.multiply(x,x), axis = 1)), [-1,1])
    u = np.matmul(x,x.T)*0.9999
    kernel = (0.5/np.pi)*(np.multiply(u, np.pi - np.arccos(u)) + np.sqrt(1.0-np.square(u)))
    return LA.eigh(kernel)

# In[ ]:

relu = lambda x: x * (x > 0) #np.maximum(x,x*0.0)
eigenvalues1, _ = kernel_eig_fast(relu)
eigenvalues1 = np.sort(eigenvalues1)
eigenvalues1 = eigenvalues1[::-1]
np.savetxt('output/relu_eig.txt', eigenvalues1)


eigenvalues1_analytical, _ = relu_analytical()
eigenvalues1_analytical = np.sort(eigenvalues1_analytical)
eigenvalues1_analytical = eigenvalues1_analytical[::-1]
np.savetxt('output/relu_eig_analytical.txt', eigenvalues1_analytical)

# In[ ]:
sigma_1 = lambda x: x * (x > 0) - (x-1) * (x > 1) #(np.maximum(x,x*0.0)-np.maximum(x-1.0,x*0.0))
eigenvalues2, _ = kernel_eig_fast(sigma_1)
eigenvalues2 = np.sort(eigenvalues2)
eigenvalues2 = eigenvalues2[::-1]
np.savetxt('output/sigma1_eig.txt', eigenvalues2)


# In[ ]:


exp = lambda x: np.exp(-0.5*np.multiply(x,x))
eigenvalues, _ = kernel_eig_fast(exp)
eigenvalues = np.sort(eigenvalues)
eigenvalues = eigenvalues[::-1]
np.savetxt('output/exp_eig.txt', eigenvalues)


# In[ ]:


sigmoid = lambda x: 1 / (1 + np.exp(-x))
eigenvalues3, _ = kernel_eig_fast(sigmoid)
eigenvalues3 = np.sort(eigenvalues3)
eigenvalues3 = eigenvalues3[::-1]
np.savetxt('output/sigmoid_eig.txt', eigenvalues3)


# In[ ]:


tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
eigenvalues4, _ = kernel_eig_fast(tanh)
eigenvalues4 = np.sort(eigenvalues4)
eigenvalues4 = eigenvalues4[::-1]
np.savetxt('output/tanh_eig.txt', eigenvalues4)


from scipy import special

def erf_analytical():
    N = 20000
    A = np.zeros((N,N))
    x = np.random.normal(0, 1, size=(N, dimension))
    x = x/np.reshape(np.sqrt(np.sum(np.multiply(x,x), axis = 1)), [-1,1])
    u = np.matmul(x,x.T)*0.9999
    kernel = (2/np.pi)*np.arctan(np.divide(2*u,np.sqrt(9-4*np.square(u))))
    return LA.eigh(kernel)

erf = lambda x: special.erf(x)
eigenvalues5, _ = kernel_eig_fast(erf)
eigenvalues5 = np.sort(eigenvalues5)
eigenvalues5 = eigenvalues5[::-1]
np.savetxt('output/erf_eig.txt', eigenvalues5)

eigenvalues5_analytical, _ = erf_analytical()
eigenvalues5_analytical = np.sort(eigenvalues5_analytical)
eigenvalues5_analytical = eigenvalues5_analytical[::-1]
np.savetxt('output/erf_eig_analytical.txt', eigenvalues5_analytical)

# In[ ]:


# Import necessary libraries
import seaborn as sns
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues1[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues1[:300]), [-1,1]))
plt.legend(title='relu\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_relu.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues1_analytical[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues1_analytical[:300]), [-1,1]))
plt.legend(title='relu analytical\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_relu_analytical.png', dpi=400, bbox_inches='tight')
plt.show()


dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues2[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues2[:300]), [-1,1]))
plt.legend(title='σ_1\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_sigma1.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues[:300]), [-1,1]))
plt.legend(title='Gaussian\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_gauss.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues3[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues3[:300]), [-1,1]))
plt.legend(title='sigmoid\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_sigmoid.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues4[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues4[:300]), [-1,1]))
plt.legend(title='tanh\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_tanh.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues5[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues5[:300]), [-1,1]))
plt.legend(title='erf\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_erf.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,301)), 'log(eig)': np.log(eigenvalues5_analytical[:300])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,301)), [-1,1]),                              np.reshape(np.log(eigenvalues5_analytical[:300]), [-1,1]))
plt.legend(title='erf analytical\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_erf_analytical.png', dpi=400, bbox_inches='tight')
plt.show()

# In[ ]:





