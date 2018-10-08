import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from RegressionMethods.OrdinaryLeastSquares import ols
from RegressionMethods.RidgeRegression import RidgeRegression
from RegressionMethods.Lasso import Lasso
from Analysis.bootstrap import bootstrap

# Load the terrain
terrain1 = imread('rjukan.tif')
# Show the terrain
plt.figure()
plt.title('Terrain Rjukan area')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Choose a smaller part of the data set
terrain = terrain1[1900:2150, 0:250]
# Show the terrain
plt.figure()
plt.title('Terrain Rjukan area')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Make zero matrix to later fit data
num_rows, num_cols = np.shape(terrain)
num_observations = num_rows * num_cols
X = np.zeros((num_observations, 3))

# make a matrix with all the values from the data on the form [x y z]
index = 0
for i in range(0, num_rows):
    for j in range(0, num_cols):
        X[index, 0] = i  # x
        X[index, 1] = j  # y
        X[index, 2] = terrain[i, j]  # z
        index += 1

############################################################################################
# OLS example
# extract x, y, z
xt = X[:,0, np.newaxis]
yt = X[:,1, np.newaxis]
zt = X[:,2, np.newaxis]

# Try the OLS-method with degree=d
d = 8
beta = ols(xt, yt, zt, degree=d)

M_ = np.c_[xt, yt]
poly = PolynomialFeatures(d)
M = poly.fit_transform(M_)
z_predict = M.dot(beta)

T = np.zeros([num_rows, num_cols])
index = 0

# create matrix for imshow
for i in range(0, num_rows):
    for j in range(0, num_cols):
        T[i, j] = (z_predict[index])
        index += 1
plt.figure()
plt.imshow(T, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Evaluate model with bootstrap algorithm
MSE, R2, bias, variance = bootstrap(xt, yt, zt, p_degree=d, method='OLS', n_bootstrap=10)
print('{0:5f} & {1:5f} & {2:5f} & {3:5f}'.format(MSE, R2, bias, variance))

# Repeat for Ridge and Lasso