import numpy as np
import matplotlib.pyplot as plt


####### Generate Training Data ###########################################################################################
N = 1000 # # of training points (training data)
degree_p = 15
penalty = 0.0000001
rn = np.random.uniform(0, 1, N)
noise = np.random.normal(0, 0.01, N)
x = np.reshape(np.sort(rn, axis=0), (N, 1))

degree_list = [i for i in range(1, degree_p)]
mu_list = np.reshape([x[i] for i in range(1, len(x))], (N-1,1))
print(mu_list)
def get_y(x, noise):
    return [ float((np.sin(4*np.pi*x[i]) + noise[i])) for i in range(len(x))]
true_y = np.array(get_y(x, noise))

# Given Basis Functions
def gaussian_basis_fn(x, mu, sigma=0.1):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)
def polynomial_basis_fn(x, degree):
    return x ** degree
def make_design(x, basisfn, basisfn_locs=None): # basisfn_locs is acts as other necessary i/p to basisfunctions
    if basisfn_locs is None:
        return np.concatenate([np.ones(x.shape), basisfn], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + \
        [basisfn(x, loc) for loc in basisfn_locs], axis=1)

# Determining Indivindual Design Matrices
design_A = make_design(x, polynomial_basis_fn, degree_list)
design_B = make_design(x, gaussian_basis_fn, mu_list)

# Generate Weights using alternative matrix equation given in section 2.2, page 2, footnote 2 as is independant of degree_p 
def weight_gen(matrix, N, penalty, y):
    return  (matrix.T).dot( np.linalg.inv(( matrix.dot(matrix.T) + (penalty*np.identity(N)) )) ).dot(y)
w_A = weight_gen(design_A, N, penalty, true_y)
w_B = weight_gen(design_B, N, penalty, true_y)
####### Generate Test Data ###########################################################################################
N=15 # # of test pts
rn = np.random.uniform(0, 1, N)
x_test = np.reshape(np.sort(rn, axis=0), (N, 1))
mu_list = [x[i] for i in range(1, len(x))] 
degree_list = [i for i in range(1, degree_p)]

design_A_test = make_design(x_test, polynomial_basis_fn, degree_list)
design_B_test = make_design(x_test, gaussian_basis_fn, mu_list)

# match test data to w_n values from training data
y_hat_A = design_A_test.dot(w_A) 
y_hat_B = design_B_test.dot(w_B)
plt.plot(x_test, y_hat_A, color='y')
plt.plot(x_test, y_hat_B, color='r')

# adding true sine
plt_x = np.arange(0,1,0.01)
plt.plot(plt_x, np.sin(4*np.pi*plt_x),color='orange')
plt.show()

