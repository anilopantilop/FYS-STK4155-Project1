from Analysis import FrankeFunction, MeanSquaredError, R2, betaConfidenceInterval_Ridge
from RidgeRegression import RidgeRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from bootstrap import bootstrap
"""
    Analysis of a Ridge Regression model of Franke's function, using set of 1000 random x and y points
"""
# Load random data, 1000 points
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]

# Calculate Franke's function without noise
z = FrankeFunction(x, y, noise=0)

########################################################################################################################
# Study dependence on lambdas
lambdas = [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]
lambdas_log = [-7, -6, -5, -4, -3, -2, -1, 0]

print('\nINVESTIGATE LAMBDAS')
Bs = []
for la in lambdas:
    Bs.append(RidgeRegression(x, y, z, l=la))

# Generate test data
x_test = np.random.rand(200)
y_test = np.random.rand(200)
z_test = FrankeFunction(x_test, y_test, noise=0)

# Calculate MSE, R2scores
M_ = np.c_[x_test, y_test]
poly5 = PolynomialFeatures(5)
M = poly5.fit_transform(M_)

MSEs = []
R2s = []
for i in range(len(lambdas)):
    z_predict = M.dot(Bs[i])
    MSE = MeanSquaredError(z_test, z_predict)
    MSEs.append(MSE)
    R2_score = R2(z_test, z_predict)
    R2s.append(R2_score)
    print('--- Lambda value: {0} ---\n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(lambdas[i], MSE, R2_score))

# Make figure
fig, ax1 = plt.subplots()
ax1.plot(lambdas_log, MSEs, 'bo-')
ax1.set_xlabel('Logarithmic lambda')
ax1.set_ylabel('MSE', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(lambdas_log, R2s, 'r*-')
ax2.set_ylabel('R2 score', color='r')
ax2.tick_params('y', colors='r')

plt.grid(True)
plt.title('Influence of lambda on MSE and R2 Score')
fig.tight_layout()
plt.show()

########################################################################################################################
# consider how the strength of noise affects the lambda values
noise = np.arange(0.001, 0.4, 0.005)
lambdas = [10**-5, 10**-4, 10**-3]
Bs = []

# Generate more data to test
x_test = np.random.rand(200)
y_test = np.random.rand(200)
M_ = np.c_[x_test, y_test]
poly5 = PolynomialFeatures(5)
M = poly5.fit_transform(M_)

for la in lambdas:
    B = []
    for n in noise:
        z = FrankeFunction(x, y, noise=n)
        B.append(RidgeRegression(x, y, z, l=la))
    Bs.append(B)

lines = []
plt.figure()
for i in range(len(lambdas)):
    print('--- lambda value:', lambdas[i], '--')
    line = []
    for j in range(len(noise)):
        z_test = FrankeFunction(x_test, y_test, noise=noise[j])
        z_predict = M.dot(Bs[i][j])
        MSE = MeanSquaredError(z_test, z_predict)
        line.append(MSE)
        R2_score = R2(z_test, z_predict)
        print(' Noise: {0} \n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(noise[j], MSE, R2_score))
    plt.plot(noise, line, label='Lambda = {0}'.format(lambdas[i]))

plt.legend()
plt.xlabel('D (Strength of noise)')
plt.ylabel('MSE')
plt.grid(True)
plt.title('Lambda and Noise')
plt.show()
########################################################################################################################
# Find the beta confidence intervals
# Load random data, 1000 points
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]

# Calculate Franke's function
z = FrankeFunction(x, y, noise=0.1)

# Generate test data
x_test = np.random.rand(1000)
y_test = np.random.rand(1000)
z_test = FrankeFunction(x_test, y_test, noise=0.1)

# Calculate beta values and polynomial matrix
beta = RidgeRegression(x, y, z, degree=5, l=10**-4)
M_ = np.c_[x_test, y_test]
poly5 = PolynomialFeatures(5)
M = poly5.fit_transform(M_)

# Calculate beta confidence intervals
conf1, conf2 = betaConfidenceInterval_Ridge(z_test, beta, M, l=10**-4)

for i in range(len(conf1)):
    print('Beta {0}: {1:5f} & [{2:5f}, {3:5f}]'.format(i, beta[i], conf1[i], conf2[i]))


########################################################################################################################
# Use bootstrap algorithm to estimate MSE, R2, bias and variance
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]
z = FrankeFunction(x, y, noise=0.1)

MSE, R2_b, bias, variance = bootstrap(x, y, z, method='Ridge', p_degree=5)
print('--- BOOTSTRAP ---')
print('MSE: ', MSE)
print('R2: ', R2_b)
print('Bias: ', bias)
print('Variance: ', variance)



