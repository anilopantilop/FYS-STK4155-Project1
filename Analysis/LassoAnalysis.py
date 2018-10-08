from Analysis import MeanSquaredError, FrankeFunction, R2
from Lasso import Lasso
from bootstrap import bootstrap
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
"""
    Analysis of a Lasso Regression model of Franke's function, using set of 1000 random x and y points
"""

# Load random data, 1000 points
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]

############################################
#   Investigate difference alphas          #
############################################
alphas = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3]
alpha_logs = [-10, -9, -8, -7, -6, -5, -4, -3]

z = FrankeFunction(x, y, noise=0)

Bs = []
for al in alphas:
    Bs.append(Lasso(x, y, z, degree=5, a=al))

# Generate new test data
x_test = np.random.rand(200)
y_test = np.random.rand(200)
z_test = FrankeFunction(x_test, y_test, noise=0)

M_ = np.c_[x_test, y_test]
poly = PolynomialFeatures(5)
M = poly.fit_transform(M_)
MSEs = []
R2s = []
for i in range(len(alphas)):
    z_predict = M.dot(Bs[i])
    MSE = MeanSquaredError(z_test, z_predict)
    MSEs.append(MSE)
    R2_score = R2(z_test, z_predict)
    R2s.append(R2_score)
    print('--- Alpha value: {0} ---\n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(alphas[i], MSE, R2_score))

fig, ax1 = plt.subplots()
ax1.plot(alpha_logs, MSEs, 'bo-')
ax1.set_xlabel('Logarithmic lambda')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('MSE', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(alpha_logs, R2s, 'r*-')
ax2.set_ylabel('R2 score', color='r')
ax2.tick_params('y', colors='r')

plt.grid(True)
plt.title('Influence of alpha on MSE and R2 Score')
fig.tight_layout()
plt.show()

########################################################################################################################
# Investigate how the alpha values are influenced by noise
noise = np.arange(0.001, 0.4, 0.01)
alphas = [10**-7, 10**-6, 10**-5, 10**-4]
Bs = []

# Generate more data to test
x_test = np.random.rand(200)
y_test = np.random.rand(200)
M_ = np.c_[x_test, y_test]
poly5 = PolynomialFeatures(5)
M = poly5.fit_transform(M_)

for al in alphas:
    B = []
    print(al)
    for n in noise:
        z = FrankeFunction(x, y, noise=n)
        B.append(Lasso(x, y, z, degree=5, a=al))
    Bs.append(B)

lines = []
plt.figure()
for i in range(len(alphas)):
    print('--- lambda value:', alphas[i], '--')
    line = []
    for j in range(len(noise)):
        z_test = FrankeFunction(x_test, y_test, noise=noise[j])
        z_predict = M.dot(Bs[i][j])
        MSE = MeanSquaredError(z_test, z_predict)
        line.append(MSE)
        R2_score = R2(z_test, z_predict)
        print(' Noise: {0} \n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(noise[j], MSE, R2_score))
    plt.plot(noise, line, label='Alpha = {0}'.format(alphas[i]))

plt.legend()
plt.xlabel('D (Strength of noise)')
plt.ylabel('MSE')
plt.grid(True)
plt.title('Alpha and Noise')
plt.show()

########################################################################################################################
# Evaluate model with bootstrap
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]
z = FrankeFunction(x, y, noise=0.1)

MSE, R2_b, bias, variance = bootstrap(x, y, z, method='Lasso', p_degree=5)
print('--- BOOTSTRAP ---')
print('MSE: ', MSE)
print('R2: ', R2_b)
print('Bias: ', bias)
print('Variance: ', variance)


