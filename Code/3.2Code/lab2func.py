import numpy as np
import matplotlib.pyplot as plt
import data_ext
import jax.numpy as jnp
from jax import random
from jax import grad

def softmax_prob(W, inputs):
    datalen=jnp.shape(inputs)[1]
    c=len(W)
    inputs= jnp.concatenate((jnp.ones((1,datalen)), inputs), axis=0)
    score= jnp.dot(W,inputs)
    large= jnp.max(score, axis=0)
    large_offset = jnp.dot(np.ones((c, datalen)),jnp.diag(large))
    expscore = jnp.exp(score-large_offset)
    norm_factor = jnp.diag(1/jnp.sum(expscore, axis=0))
    return jnp.dot(expscore, norm_factor).T
def final_softmax_prob(W, inputs):
    datalen=jnp.shape(inputs)[1]
    c=len(W)
    inputs= jnp.concatenate((jnp.ones((1,datalen)), inputs), axis=0)
    print(W)
    score= jnp.dot(W,inputs)
    large= jnp.max(score, axis=0)
    large_offset = jnp.dot(np.ones((c, datalen)),jnp.diag(large))
    expscore = jnp.exp(score-large_offset)
    norm_factor = jnp.diag(1/jnp.sum(expscore, axis=0))
    return jnp.dot(expscore, norm_factor).T

def softmax_xentropy(Wb, inputs, targets, num_classes):
    epsilon = 1e-8
    ys = get_one_hot(targets, num_classes)
    logprobs = -jnp.log(softmax_prob(Wb, inputs)+epsilon)
    return jnp.mean(ys*logprobs)
def get_one_hot(targets, num_classes):
    res= jnp.eye(num_classes)[jnp.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[num_classes])

def grad_descent(Wb, inputs, targets, num_classes,  lrate, nsteps):
    W1=Wb
    Whist=[W1]
    losshist=[softmax_xentropy(W1,inputs, targets, num_classes)]
    eta=lrate
    for i in range(nsteps):
        gWb=grad(softmax_xentropy,(0))(W1, inputs, targets, num_classes)
        W1=W1-eta*gWb
        if(i%5==0):
            Whist.append(W1)
            losshist.append(softmax_xentropy(W1, inputs, targets, num_classes))
    Whist.append(W1)
    g = softmax_xentropy(W1, inputs, targets, num_classes)
    losshist.append(g)
    return W1, Whist, losshist
