import numpy as np
import matplotlib.pyplot as plt
import data_ext
from jax import random

####### Extract Data ###########################################################################################
META = data_ext.returnMETADATA('iris.data') # return dict of data

dim = 4

N_c = len(META)
N = 150
class_name = []
for i in META:
    class_name.append(i)
class_data = [[] for i in range(len(class_name))]
class_mean = [[] for i in range(len(class_name))]
class_cov = [[] for i in range(len(class_name))]
for i in META: # sift through dictionary of METADATA
    for ID in META[i]: # loop through id of flowers
        temp = []
        for dim in META[i][ID]: 
            temp.append( float(META[i][ID][dim]) )
        temp = np.reshape(temp, (1,len(temp))) # 4 dim matrix that represents one point in iris data
        class_data[list(META).index(i)].append(temp) # assign matrix to specific class data

####### Functions ####### SUM (n_c/N) (u_c - u)(u_c - u)T
def softmax_prob(W, inputs):
    datalen=jnp.shape(inputs)[1]
    c=len(W)
    inputs= np.concatenate((jnp.ones((1,datalen)), inputs), axis=0)
    score= np.dot(W,inputs)
    large= np.max(score, axis=0)
    large_offset= np.dot(np.ones((c, datalen)),np.diag(large))
    expscore=np.exp(score-large_offset)
    norm_factor=np.diag(1/np.sum(expscore, axis=0))
    return np.dot(expscore, norm_factor).T

def softmax_xentropy(Wb, inputs, targets, num_classes):
    epsilon= 1e-8
    ys=get_one_hot(targets, num_classes)
    logprobs= -np.log(softmax_prob(Wb, inputs)+epsilon)
    return np.mean(ys*logprobs)
def get_one_hot(targets, num_classes):
    res= np.eye(num_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[num_classes])

Wb=np.array([[-3.,1.3,2.0,-1.0], [-6.,-2.,-3.,1.5], [1.,2.0,2.0,2.5], [3.,4.0,4.0,-2.5]])
inputs=np.array([[0.52,1.12,0.77],[3.82,-6.11,3.15],[0.88,-1.08,0.15],[0.52,0.06,-1.30],[0.74,-2.49,1.39],[0.14,-0.43,-1.69]]).T

targets=np.array([0,1,3,2,1,2])

key= randomseed(0)
key, W_key=random.split(key,2)
[classes, dim]=4,3
Winit= np.random.normal(W_key, (classes, dim+1))

def grad_descent(Wb, inputs, targets, num_classes,  lrate, nsteps):
    W1=Wb
    Whist=[W1]
    losshist=[softmax_xentropy(W1,inputs, targets, num_classes )]
    eta=lrate
    for i in range(nsteps):
        gWb=grad(softmax_xentropy, (0))(W1, inputs, targets, num_classes)
        W1=W1-eta*gWb
        if(i%5==0):
            Whist.append(W1)
            losshist.append(softmax_xentropy(W1, inputs, targets, num_classes))
    Whist.append(W1)
    losshist.append(softmax_xentropy(W1, inputs, targets, num_classes))
    return W1, Whist, losshist

W2, Whist, losshist=grad_descent(Winit, inputs, targets,4,0.75,200)
plt.plot([5*iforiinrange(len(losshist))], losshist)

"""
plt.legend(loc='upper right')
plt.ylabel('Frequency of Value')
plt.xlabel('Projection Values')
"""
plt.show()
