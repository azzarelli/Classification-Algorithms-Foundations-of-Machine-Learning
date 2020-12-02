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
####### Solving Eigenvalue Problem with known index of best eigenvector (which reduces info loss for projections) ########
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
w_star = eig_vecs[:,0].reshape(4,1).real # known from 3_2_1
eigval = eig_vals[0]

####### Projection calculation #######
y_proj = [[] for i in range(N_c)]

for i in range(N_c):
    for j in range(len(class_data[i])):
        dat = (class_data[i][j].reshape((4,1)))
        y = float((w_star.T).dot(dat))
        y_proj[i].append(y)

####### Plot Histogram #######
bins = 50
print(y_proj[0])
plt.hist(y_proj[0], bins, facecolor='g', alpha=0.5, label=class_name[0])
plt.hist(y_proj[1], bins, facecolor='r', alpha=0.5, label=class_name[1])
plt.hist(y_proj[2], bins, facecolor='b', alpha=0.5, label=class_name[2])

plt.legend(loc='upper right')
plt.ylabel('Frequency of Value')
plt.xlabel('Projection Values')
plt.show()
