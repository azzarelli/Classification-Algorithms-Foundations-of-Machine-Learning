import numpy as np
import matplotlib.pyplot as plt

####### General Constant #######
N_a = 25 # assign size of data points for each class
N_b = N_a
degree_p = 10
penalty = 0.0000001

pt1 = [5,5] # a random point which we want to classify

####### Generating Distributions ###########################################################################################
mean_a, mean_b = np.reshape([1,1],(2,)) , np.reshape([5,5],(2,)) # assign mean
cov_a, cov_b = np.reshape([[6, 0],[0, 3]], (2,2)), np.reshape([[4, 1],[1, 3]], (2,2)) # assign covariance(.)

x_a = np.random.multivariate_normal(mean_a, cov_a, N_a) # distribute data over multivariate gauss (normal) distribution
x_b = np.random.multivariate_normal(mean_b, cov_b, N_b)

####### Log-Odds Calculation (Assume class-size-priors) ###########################################################################################

## Using Estimated Parameters ##
mu_a_est = [sum(x_a[:,0])/N_a , sum(x_a[:,1])/N_a]
mu_b_est = [sum(x_b[:,0])/N_b , sum(x_b[:,1])/N_b]
mu_a_est = np.array(mu_a_est)
mu_b_est = np.array(mu_b_est)
SUMA = [[0,0],[0,0]] 
for i in range(N_a): # estimating cov(.) using  SUM( (x_i - mu)(x_i - mu)^T )/N
    s = x_a[i]- mu_a_est
    SUMA = [[SUMA[0][0] + s[0]**2, SUMA[0][1] + s[0]*s[1]], [SUMA[1][0] + s[0]*s[1], SUMA[1][1] + s[1]**2]]
SUMB = [[0,0],[0,0]]
for i in range(N_b):
    s = x_b[i]- mu_b_est
    SUMB = [[SUMB[0][0] + s[0]**2, SUMB[0][1] + s[0]*s[1]], [SUMB[1][0] + s[0]*s[1], SUMB[1][1] + s[1]**2]]
cov_a_est = np.array(SUMA)/N_a
cov_b_est = np.array(SUMB)/N_b

L_a = np.linalg.inv(cov_a_est) # precision matrix for class a, b
L_b = np.linalg.inv(cov_b_est)

def decision_bound_lin(ma,mb, cov, x): # use the equation 2(u1-u2)^(T).S^(-1).x + u2^(T).L.u2 - u1^(T).L.u1 - 2ln(P(Ca)/P(Cb)) = 0, where prior=0
                                        # and rearrange for y= f(x)
    va, wa, vb, wb = ma[0], ma[1], mb[0], mb[1]  
    s,t = ma[0]-mb[0], ma[1]-mb[1] # u1 - u2
    a = cov[0][0]
    b = cov[0][1]
    c = cov[1][0]
    d = cov[1][1]
    pt3a = ((va*va*a)+(va*c*wa)+(va*b*wa)+(wa*wa*d)) # u1^(T).L.u1 
    pt3b = ((vb*vb*a)+(vb*c*wb)+(vb*b*wb)+(wb*wb*d)) # u2^(T).L.u2
    # equation format y = Mx + C
    C = (pt3b-pt3a)/((2*b*s)+(2*d*t))
    M = ((a*s)+(c*t))/((b*s)+(d*t))      
    return -x*M - C

x_1 = [i for i in np.arange(-5,10,0.2)]
x_2 = [decision_bound_lin(mu_a_est, mu_b_est, L_a, x_1[i]) for i in range(len(x_1))] # Linear equation used for determining decision boundary

#plt.plot(x_2, x_1, color='green') # Linear decision boundary plot

def LogOdds(pt1, cov_a, cov_b, mean_a, mean_b): # use these values to determine the Log-odds of a singular point
    L_a = np.linalg.inv(cov_a) # precision matrix for class a, b
    L_b = np.linalg.inv(cov_b)
    Q_b = 0.5*((pt1-mean_b).T.dot(L_b).dot(pt1-mean_b)) # Quad form of G.distribution for class a,b
    Q_a = 0.5*((pt1-mean_a).T.dot(L_a).dot(pt1-mean_a))
    det_L_a = np.linalg.det(L_a)
    det_L_b = np.linalg.det(L_b)
    return np.log(det_L_a/det_L_b)+Q_b-Q_a # taken from equation for 2-class logodss noprior classification

def LogOdds_grid(size, X, Y, cov_a, cov_b, mean_a, mean_b): # mesh-grid method to plot quadratic decision boundary
    RANGE = np.arange(size[0], size[1], 0.05) # 
    z = [] # set array to append coordinates that make up the quadratic boundary
    for  i in range(len(X)): # y axis iteration
        for j in range(len(Y)): # x axis iteration
            pt1 = [X[j][i], Y[j][i]]
            LO = LogOdds( pt1,cov_a, cov_b, mean_a, mean_b ) # use log-odds function to determine the value for point[j][i] in mesh
            if (LO < 0.01) and (LO > -0.01): # can't us '== 0' as the mesh isn't "fine" enough to result in a log-odds value of 0, so around 0 is enough to give an estimation for DB 
                z.append(pt1)
    return z 

size = [-10,10] # bounds for mesh
x = np.arange(size[0], size[1], 0.1) # sqaure meshgrid
y = np.arange(size[0], size[1], 0.1)
X, Y = np.meshgrid(x, y) # generate mesh
Z = LogOdds_grid(size,X,Y, cov_a_est, cov_b_est, mean_a, mean_b)
z = np.reshape(Z, (len(Z), 2)) # use mesh to determine areas where 

plt.plot(z[:,0],z[:,1],color='green') # plot d-b

plt.plot(mu_a_est[0], mu_a_est[1], '*', color='blue') # plot esitmated mean
plt.plot(mu_b_est[0], mu_b_est[1], '*', color='red')
plt.plot(x_a[:,0],x_a[:,1],'.',color='blue') # plot class distribution
plt.plot(x_b[:,0],x_b[:,1],'.',color='red')
plt.show()
