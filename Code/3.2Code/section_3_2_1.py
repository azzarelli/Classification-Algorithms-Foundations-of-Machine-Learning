import numpy as np
import matplotlib.pyplot as plt
import data_ext

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
for i in META: # sift through dictionary of METADATA
    for ID in META[i]: # loop through id of flowers
        temp = []
        for dim in META[i][ID]: 
            temp.append( float(META[i][ID][dim]) )
        temp = np.reshape(temp, (1,len(temp))) # 4 dim matrix that represents one point in iris data
        class_data[list(META).index(i)].append(temp) # assign matrix to specific class data

for i in range(len(class_data)):
    class_mean[i] = np.mean(class_data[i], axis=0) # estimating class means
    
mu_of_mean =  np.mean(class_mean, axis=0) # calculating mean of classmeans

####### Between-class Cov ####### SUM (n_c/N) (u_c - u)(u_c - u)T
SUM = np.zeros((4, 4))
for i in range(len(class_mean)):
    m_c, m = class_mean[i].reshape(4,1), mu_of_mean.reshape(4,1)
    diff = m_c - m
    SUM += (diff.dot(diff.T))*(50) # equation for between class cov given in coursework
S_b = SUM

####### Within-class Cov ####### SUM  ClassCov
S_w = np.zeros((4,4))
for i in range(len(class_mean)):
    class_mat = np.zeros((4,4)) # scatter matrix for every class
    for x in class_data[i]:
        x, m_c = x.reshape(4,1), class_mean[i].reshape(4,1) # make column vectors
        class_mat += (x-m_c).dot((x-m_c).T)
    S_w += class_mat 
####### Solving Eigenvalue Problem ########
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
eigvec = eig_vecs[:,0].reshape(4,1) # iris data is fixed, I have taken eigenvectors/values that I evaluated using the loop below, and ensured  S_b.w = l*S_w.w  is satisfied
eigval = eig_vals[0]

""" Rank Condition
print(np.linalg.matrix_rank(S_b))
The number of lineasr distriminants is constrained by < c-1, where c = # of classes.
In the case where rank = 1, there exists perfect collinearity and covariance matrix would have rank 1 (i.e. one eigen vector with a non-zero vale).
Here rank = 2, we should find two solutions, in this case it is present at position [0],[1] of eig_vals and eig_vecs. I know this because I ran the below
loop and verified which values satisfied S_b.w = l*S_w.w

for i in range(len(eig_vals)):
    eigvec = eig_vecs[:,i].reshape(4,1)
    eigval = eig_vals[i]
    print(S_b.dot(eigvec))
    print(eigval*S_w.dot(eigvec),"\n")
"""

####### Optimising Fisher Score ########
"""
I am going to use explained percentage to reduce information loss dependant on my eigenvector (optimal direction w*) choice
We are going from 4-D space into 1-D space when projecting points onto y, this is a large drop in dimensionality and therefore
it is imperative to reduce information loss. Using explained percentage will allow me to select the eigenvec (w*) with the least information loss
This ideas was taken from https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html,
the LDA explained variance componenet
"""
eigvecs = [eig_vecs[:,0].reshape(4,1), eig_vecs[:,1].reshape(4,1)] # I know the positions of the non-zero eigenvalues and related eigenvectors, [0, 1]
eigvals = [eig_vals[0], eig_vals[1]]

eigv_sum = sum(eig_vals)
for i in range(len(eigvals)):
    print('eigenvalues: ', ((eigvals[i]/eigv_sum).real)*100)

"""
It is evident that eigen value and corresponding vecotr at position [0] save the most information. We will use this.
"""
w_star = eigvecs[0]
print("\nChosen w*: \n", w_star.real)


