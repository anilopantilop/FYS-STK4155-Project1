import unittest
import numpy as np
from Analysis.bootstrap import bootstrap
from Analysis.Analysis import FrankeFunction, R2
from RegressionMethods.RidgeRegression import RidgeRegression
from RegressionMethods.Lasso import Lasso


class TestAnalysisMethods(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(200)
        self.y = np.random.rand(200)
        self.z = FrankeFunction(self.x, self.y, noise=0)

    def testBootstrap(self):
        """ Doesn't accept no method input """
        self.assertEqual(0, bootstrap(self.x, self.y, self.z, p_degree=5, method='No METHOD', n_bootstrap=100))

    def testR2(self):
        """ Test R2 less than 1"""
        z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        z_hat = np.array([2, 2, 4, 5, 6, 7, 8, 9, 10])
        self.assertLess(R2(z, z_hat), 1)


    def testRidgeOutput(self):
        """ Test beta is the correct shape """
        beta = RidgeRegression(self.x, self.y, self.z, degree=5)
        (M,) = np.shape(beta)
        self.assertEqual(M, 21)

        beta2 = RidgeRegression(self.x, self.y, self.z, degree=2)
        (M2,) = np.shape(beta2)
        self.assertEqual(M2, 6)

    def testLassoOutput(self):
        """ Test beta is the correct shape """
        beta = Lasso(self.x, self.y, self.z, degree=5)
        (M,) = np.shape(beta)
        self.assertEqual(M, 21)

        beta2 = Lasso(self.x, self.y, self.z, degree=2)
        (M2,) = np.shape(beta2)
        self.assertEqual(M2, 6)