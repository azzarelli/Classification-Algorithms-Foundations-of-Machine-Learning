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

####### Edited sklearn PCA example ########
####### LINK: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
from sklearn.decomposition import PCA

X = training
targets=jnp.array([int(i/50) for i in range(150)])

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

plt.figure()
colors = ['green', 'red', 'blue']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], class_name):
    plt.scatter(X_r[targets == i, 0], X_r[targets == i, 1], color=color, alpha=.8, lw=lw,
                label=class_name[i])
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

####### plotting porjections ########
test = jnp.reshape(jnp.array([4,4,1,0.5]), (1,4)) # non-T test data
transf = pca.transform(test) # generate 2D weights for projections
score = pca.explained_variance_ratio_
print(score)
plt.scatter(transf[0][0], transf[0][1], color='black')

plt.show()
