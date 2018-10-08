from Analysis import plotFrankes, MeanSquaredError, R2, FrankeFunction, var2, varBeta, betaConfidenceInterval_OLS, bias, var_f
from OrdinaryLeastSquare import ols
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from kFoldValidation import k_fold_validation



# Evaluate model with bootstrap
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]
z = FrankeFunction(x, y, noise=0.1)

MSE, R2_b, bias, variance = bootstrap(x, y, z, method='OLS', p_degree=5)
print('--- BOOTSTRAP for OLS ---')
print('MSE: ', MSE)
print('R2: ', R2_b)
print('Bias: ', bias)
print('Variance: ', variance)


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
conf1, conf2 = betaConfidenceInterval_OLS(z_test, beta, M)

for i in range(len(conf1)):
    print('Beta {0}: {1:5f} & [{2:5f}, {3:5f}]'.format(i, beta[i], conf1[i], conf2[i]))
