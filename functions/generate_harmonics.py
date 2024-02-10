#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#import spherical_harmonics.tensorflow  # run computation in TensorFlow

from spherical_harmonics import SphericalHarmonics
from spherical_harmonics.utils import l2norm

dimension = 3
degree = 22
# Returns all the spherical harmonics in dimension 3 up to degree 10.

Phi = SphericalHarmonics(dimension, [degree])

x = np.random.randn(10000, dimension)  # Create random points to evaluation Phi
x = x / l2norm(x)  # normalize vectors
out = Phi(x)  # Evaluate spherical harmonics at `x`


# In[2]:


w = np.random.randn(out.shape[1])
w = w/ l2norm(w)
res = np.matmul(out, np.reshape(w, [-1,1]))


# In[3]:


res.shape


# In[4]:


np.savetxt('dim{}_k{}_x.txt'.format(dimension,degree), x)
np.savetxt('dim{}_k{}_y.txt'.format(dimension,degree), res)


# In[ ]:





# In[ ]:




