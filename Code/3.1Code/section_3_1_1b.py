import numpy as np
import matplotlib.pyplot as plt
import random

####### Generate Training Data ###########################################################################################
degree_p = 10
penalty = 0.0000001

mean_a, mean_b = np.reshape([1,1],(2,)) , np.reshape([3,3],(2,))

matrices = [np.reshape([[8, 2],[2, 3]], (2,2)), np.reshape([[4, 0],[0, 1]], (2,2)), np.reshape([[8, -2],[-2, 3]], (2,2))]
plt.figure(figsize=(20, 5))
pltnum = 130

for i in range(3): # four different cov(.), S_a values
    cov_a = matrices[i]
    cov_b = matrices[i]#np.reshape([[var_x_b, (alpha_b*var_x_b)],[(alpha_b*var_x_b), ((alpha_b**2)*var_x_b)+var_y_b]], (2,2))

    print(cov_a)
    print(cov_b)

    N_a = 25
    N_b = N_a

    x_a = np.random.multivariate_normal(mean_a, cov_a, N_a) # distribute data over multivariate gauss (normal) distribution
    x_b = np.random.multivariate_normal(mean_b, cov_b, N_b)

    w = np.reshape([11,-1],(2,)) # some random w

    y_a = [(w.T).dot(x_a[i]) for i in range(N_a)] # calculation of y_c^n
    y_b = [(w.T).dot(x_b[i]) for i in range(N_b)]

    mu_a = np.reshape([sum(x_a[:,0])/N_a, sum(x_a[:,1])/N_a], (2,))
    mu_b = np.reshape([sum(x_b[:,0])/N_b, sum(x_b[:,1])/N_b], (2,))
    m_a = (w.T).dot(mu_a)
    m_b = (w.T).dot(mu_b)

    s2_a = sum((y_a + m_a)**2)/N_a
    s2_b = sum((y_b + m_b)**2)/N_b

    bins = 20

    plt.subplot(131+i)
    plt.hist(y_a, bins, alpha=0.1, label='y_a')
    plt.hist(y_b, bins, alpha=0.2, label='y_b')
    plt.legend(loc='upper right')

plt.show()
