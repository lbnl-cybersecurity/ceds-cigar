
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time


# In[ ]:


ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)


# # Definition

# In[ ]:


# generate object and put it to shared store
@ray.remote
def gen_data():
    objectID = [1,2,3,4,5]
    return objectID

# create a generator class for an object on shared store
@ray.remote
class gener:
    def __init__(self, objectID):
        self.i = 0
        self.n = len(objectID)
        self.objectID = objectID

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.n:
            val = self.objectID[self.i]
            self.i += 1
            return val
        else:
            raise StopIteration()


# # Test 

# In[ ]:


a = gen_data.remote()


# In[ ]:


itera = gener.remote(a)
ray.get(itera.next.remote())
itera2 = gener.remote(a)
ray.get(itera2.next.remote())


# Try to define a generator type object, fail to pickle.  
# **Fail**

# In[ ]:


@ray.remote
def iterator(objectID):
    i = 0
    n = len(objectID)
    while i < n:
        yield objectID[i]
        i += 1


# In[ ]:


itera3 = iterator.remote(a)

