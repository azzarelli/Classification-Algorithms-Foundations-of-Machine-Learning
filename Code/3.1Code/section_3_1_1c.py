import numpy as np
import matplotlib.pyplot as plt
import random

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

w_0 = np.reshape([1,1],(2,)) # apply abritrary initial w

F = [] # set up plotting array and graph subplot
plt.subplot(131)

for k in range(100): # i (100) equally spaced steps through the rotation
    sine = np.sin((k*2*np.pi)/100)
    cosine = np.cos((k*2*np.pi)/100)
    R = np.reshape([[cosine,-sine],[sine,cosine]],(2,2)) # new Rot. matrix
    w = R.dot(w_0) # apply for new w
    y_a = [(w.T).dot(x_a[i]) for i in range(N_a)]# calculation of y_c^n
    y_b = [(w.T).dot(x_b[i]) for i in range(N_b)]

    mu_a = sum(y_a)/len(y_a) # mu_c = SUM( y_c ) / N_c
    mu_b = sum(y_b)/len(y_b)

    s2_a = sum((y_a + mu_a)**2)/N_a # sig_c = SUM( y_{c,n} - mu_c )^2 / N_c
    s2_b = sum((y_b + mu_b)**2)/N_b

    NOM = (mu_a - mu_b)**2 # nominator from cwk documentation
    DOM = ( (N_a)/(N_a+N_b) )*s2_a+( (N_b)/(N_a+N_b) )*s2_b  # denominator from cwk documentation
    F.append(NOM/DOM) # fisher ratio equation


argmax = np.sort(F) # find the larges value for F-ratio from all calculated values of F
x_index = F.index(argmax[99]) # determine index of this value
xmax = (x_index*360)/100 # detemine x-coord for max, this oscilates with a period of pi
ymax = max(F) # determine y-coord for max
print('argmax()=', ymax, " at ", xmax,"Deg")

sine = np.sin((ymax*2*np.pi)/360) 
cosine = np.cos((ymax*2*np.pi)/360)
R = np.reshape([[cosine,-sine],[sine,cosine]],(2,2)) # calculate the "best" rotation matrix
w_star = R.dot(w_0) # determine w* value from rotation matrix applied to initial w
print("w* = ", w_star)

# plot with max annotation
plt.annotate('global max', xy=(xmax, ymax), xytext=(xmax, ymax+0.015),arrowprops=dict(facecolor='black', shrink=0.01))
plt.plot([(i*360)/100 for i in range(100)], F, label='matrix', color='red')
plt.xlabel("Degress")
plt.ylabel("F(w)")
plt.show()
