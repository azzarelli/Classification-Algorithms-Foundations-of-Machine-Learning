import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import empirical_covariance
from sklearn.covariance import MinCovDet

####### Generate Gaussian Data & Distributions ###########################################################################################
degree_p = 10
penalty = 0.0000001

# mean and cov assigned values, unlike previously as the purpose of this test does not rely on specific mu and scov values
mean_a, mean_b = np.reshape([1,1],(2,)) , np.reshape([3,3],(2,)) # assign mean
cov_a, cov_b = np.reshape([[6, 2],[2, 3]], (2,2)), np.reshape([[6, 2],[2, 3]], (2,2)) # assign covariance(.)
plt.figure(figsize=(20, 5)) # set figure size
N_a = 25 # assign size of data points for each class
N_b = N_a
x_a = np.random.multivariate_normal(mean_a, cov_a, N_a) # distribute data over multivariate gauss (normal) distribution
x_b = np.random.multivariate_normal(mean_b, cov_b, N_b)

####### Determine Mu and Cov(.) ###########################################################################################
mu_a = sum(x_a)/N_a # mu_c = SUM( x_c ) / N_c, where x_c is the coordinate position of mean 
mu_b = sum(x_b)/N_b

SUM_a =  [[0,0],[0,0]] # define summation of cov to be used
SUM_b =  [[0,0],[0,0]]

for i in range(N_a):
    LHS_a = np.reshape((x_a[i] - mu_a), (1,2)) # detemine LHS and RHS of summation for sigma estimation
    RHS_a = LHS_a.T
    dot_a = LHS_a*RHS_a # combine for the summation value
    LHS_b = np.reshape((x_b[i] - mu_b), (1,2))
    RHS_b = LHS_b.T
    dot_b = LHS_b*RHS_b
    
    SUM_a +=dot_a
    SUM_b +=dot_b

sig_a = SUM_a/N_a # finally out sigma predictions 
sig_b = SUM_b/N_b

print("true (mu_c) Values: \n mean_a=", mean_a,"\n mean_b=", mean_b)
print("mean (mu_c) Predictions: \n mu_a=", mu_a,"\n mu_b=", mu_b)

print("\ntrue Covariance = \n", cov_a)
print("my prediciton = \n", sig_a)
print("sklean_empirical = \n", empirical_covariance(x_a)) # compare with sklearn empirical covariance calculation (which utlises the same equation)
mincovdet_esti = MinCovDet(random_state=0).fit(x_a)
print("sklearn MinCovDet = \n", mincovdet_esti.covariance_) # compare with mincovdet calculation (which utilises different equation)


