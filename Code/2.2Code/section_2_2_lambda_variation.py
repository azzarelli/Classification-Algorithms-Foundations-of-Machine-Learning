import numpy as np
import matplotlib.pyplot as plt
import random

split_arr = np.arange(0.1, 1, 0.2)

for k in range(len(split_arr)):
    degree_p = 10
    penalty = 0.0000001

    N_total = 1000 # # of training points (training data)
    train_split_ratio =  split_arr[k] # training split as a ratio of total data:  # of training / total # of data
    N_train = int(train_split_ratio*N_total) # this will floor if not whole number
    N_test = N_total - N_train # associate remaining total to testing
    print("Data Type | Data Points \nTraining ", N_train, "\nTesting   ", N_test)
    influence = 'random'
    # random influence - separating after sorting
    if influence == "random": 
        rn = np.random.uniform(0, 1, N_total)
        x = list(np.sort(rn, axis=0))
        rand_positions = random.sample(range(N_total), N_test) # random array of N_test(-number) positions to give random points of data associated with testing data
        sorted_positions = list(np.sort(rand_positions, axis=0))
        x_test_list = []
        for i in range(N_test):
            x_test_list.append(x[sorted_positions[i]-i]) # as you pop off values you need to compensate for the index shifting left by 1 every time hence "-i" 
            x.pop(sorted_positions[i]-i)

        x_train = np.reshape(x, (N_train, 1))
        x_test = np.reshape(x_test_list, (N_test, 1))
        print("Random Test-Train split")

    # non-random shuffle - separate before sorting
    else:
        rn_train = np.random.uniform(0, 1, N_train)
        rn_test = np.random.uniform(0, 1, N_test)
        x_train = np.reshape(np.sort(rn_train, axis=0), (N_train, 1))
        x_test =  np.reshape(np.sort(rn_test, axis=0), (N_test, 1))
        print("Seperate Random Generation")

    test_its = 30
    msr_plot_A = []
    msr_plot_B = []
    min_B = -20
    max_B = 10

    for i in range(-min_B,-max_B,-1):
        residuals_array_A = []
        residuals_array_B = []
        for j in range(8): # have to repeat and take average of >8 to be scientifically correct
            penalty = 10**-i
            degree_list = [i for i in range(1, degree_p)]
            list_ = np.sort( np.random.randint(np.size(x_train, axis=0), size=degree_p-1) )
            mu_list = [[float(x_train[list_[i]]) for i in range(degree_p-1)] for j in range(N_train)]

            noise = np.random.normal(0, 0.01, N_train)
            def get_y(x, noise):
                return [ float((np.sin(4*np.pi*x[i]) + noise[i])) for i in range(len(x))]
            true_y = np.array(get_y(x_train, noise))

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
            design_A = make_design(x_train, polynomial_basis_fn, degree_list)
            design_B = make_design(x_train, gaussian_basis_fn, mu_list)

            # Generate Weights using alternative matrix equation given in section 2.2, page 2, footnote 2 as is independant of degree_p 
            def weight_gen(matrix, N, penalty, y):
                return  (matrix.T).dot( np.linalg.inv(( matrix.dot(matrix.T) + (penalty*np.identity(N)) )) ).dot(y)
            w_A = weight_gen(design_A, N_train, penalty, true_y)
            w_B = weight_gen(design_B, N_train, penalty, true_y)

            design_A_test = make_design(x_test, polynomial_basis_fn, degree_list)
            design_B_test = make_design(x_test, gaussian_basis_fn, mu_list)
                
        # match test data to w_n values from training data
            y_hat_A = design_A_test.dot(w_A) 
            y_hat_B = design_B_test.dot(w_B)
            real_y = np.sin(4*np.pi*x_test).ravel()

        # calulate residuals
            sq_residuals_A = (real_y - y_hat_A)**2
            mean_sq_residual_A = sum(sq_residuals_A)/len(sq_residuals_A)
            residuals_array_A.append(mean_sq_residual_A)

            sq_residuals_B = (real_y - y_hat_B)**2
            mean_sq_residual_B = sum(sq_residuals_B)/len(sq_residuals_B)
            residuals_array_B.append(mean_sq_residual_B)
            
        msr_plot_A.append(sum(residuals_array_A)/len(residuals_array_A))
        msr_plot_B.append(sum(residuals_array_B)/len(residuals_array_B))

    msr_plot_A = np.clip(msr_plot_A,0,2)
    msr_plot_B = np.clip(msr_plot_B,0,2)
    plt.xscale("log", base = 10)
    plt.plot([float(10**-(i)) for i in range(-min_B, -max_B, -1)], msr_plot_A, label=('PBF '+"{:.2f}".format(split_arr[k])))
    #plt.plot([float(10**-(i)) for i in range(-min_B, -max_B, -1)],  color='blue', label=('PBF '+"{:.2f}".format(split_arr[k])))
# adding true sine
plt.xlabel("lambda")
plt.ylabel("MSR Value")
legend = plt.legend(loc='upper right', shadow=True)
plt.show()

