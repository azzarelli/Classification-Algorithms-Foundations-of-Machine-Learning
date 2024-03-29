import numpy as np
import matplotlib.pyplot as plt
import data_ext
import lab2func as lf
import jax.numpy as jnp
from jax import random
from jax import grad

####### Extract Data ###########################################################################################
META = data_ext.returnMETADATA('iris.data') # return dict of data

N_c = len(META)
N = 150
class_name = []
for i in META:
    class_name.append(i)
class_data = [[] for i in range(len(class_name))]
class_mean = [[] for i in range(len(class_name))]
class_cov = [[] for i in range(len(class_name))]

training = []
for i in META: # sift through dictionary of METADATA
    for ID in META[i]: # loop through id of flowers
        temp = []
        for dim in META[i][ID]: 
            temp.append( float(META[i][ID][dim]) )
        training.append(temp)
        #temp = np.reshape(temp, (1,len(temp))) # 4 dim matrix that represents one point in iris data
        class_data[list(META).index(i)].append(temp) # assign matrix to specific class data
training = jnp.array(training) # need to concat into jnp array

####### Constant Definitions ########
l_rate = 0.75 # learning rate
its = 500
inputs= training.T
targets=jnp.array([int(i/50) for i in range(150)])

key=random.PRNGKey(0)
key, W_key = random.split(key,2)
[classes, dim]= 3,4

Winit= random.normal(W_key, (classes, dim+1))

W2, Whist, losshist = lf.grad_descent(Winit, inputs, targets, classes, l_rate, its)

test = jnp.reshape(jnp.array([4,4,1,0.5]), (1,4)).T
weights =  np.around(lf.softmax_prob(W2, test),3)
print(weights)
