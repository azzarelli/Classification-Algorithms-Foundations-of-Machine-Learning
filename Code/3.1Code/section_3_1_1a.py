import numpy as np
import matplotlib.pyplot as plt
import random

####### Generate Training Data ###########################################################################################
degree_p = 10
penalty = 0.0000001

var_x_a, var_y_a, var_x_b, var_y_b, alpha_a, alpha_b = 3, float(2),3, float(2), float(-1), float(-1)

mean_a, mean_b = np.reshape([1,1],(2,)) , np.reshape([3,3],(2,))
cov_a = np.reshape([[var_x_a, (alpha_a*var_x_a)],[(alpha_a*var_x_a), ((alpha_a**2)*var_x_a)+var_y_a]], (2,2)) # calculate COV(.) for class a
cov_b = np.reshape([[var_x_b, (alpha_b*var_x_b)],[(alpha_b*var_x_b), ((alpha_b**2)*var_x_b)+var_y_b]], (2,2)) # " for class b
inv_cov_a = np.reshape([[((1/var_x_a)+(alpha_a/var_y_a)), -(alpha_a/var_y_a)],[-(alpha_a/var_y_a), ((alpha_a**2)/var_y_a)]], (2,2))
inv_cov_b = np.reshape([[((1/var_x_b)+(alpha_b/var_y_b)), -(alpha_b/var_y_b)],[-(alpha_b/var_y_b), ((alpha_b**2)/var_y_b)]], (2,2))

N_a = 25
N_b = N_a

x_a = np.random.multivariate_normal(mean_a, cov_a, N_a) # distribute data over multivariate gauss (normal) distribution
x_b = np.random.multivariate_normal(mean_b, cov_b, N_b)

w = np.reshape([11,-1],(2,)) # some random w

y_a = [(w.T).dot(x_a[i]) for i in range(N_a)] # calculation of y_c^n
y_b = [(w.T).dot(x_b[i]) for i in range(N_b)]

print(y_a)
bins = 20
plt.hist(y_a, bins, facecolor='g', alpha=0.2, label='y_a')
plt.hist(y_b, bins, facecolor='r', alpha=0.2, label='y_b')
#plt.plot(x_a[:,0],x_a[:,1], 'o', color='blue')
#plt.plot(x_b[:,0],x_b[:,1], 'o', color='red')
plt.legend(loc='upper right')
plt.ylabel('density')
plt.xlabel('y_class value')
plt.show()
