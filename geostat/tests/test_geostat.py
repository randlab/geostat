import unittest
import geostat as geo
import numpy as np

def float_eq(a, b, tolerance=1e-3):
    return abs(a-b) < tolerance


class TestOrdinaryKrigingVariogram(unittest.TestCase):

    def setUp(self):
        self.x = 1
        self.y = 2

    def test_Kriging3points1D(self):
        """  TEST METHOD

        Kriging case with three points in 1D and a linear variogram.
        Reference value: Kitanidis (1996) page 169
        """
        x = np.array([[0, 0], [1, 0], [3, 0]])
        v = np.array([1, 0, 0])
        xu = np.array([[2, 0]])
        vario = geo.Variogram(rang=30, sill=1, typ="linear", nugget=1)
        estim1, so1 = geo.ordinary_kriging_variogram(x, v, xu, vario)

        # Value of lambda1 given by Kitanidis on page 169
        estim1_solution = 0.1304

        self.assertTrue(float_eq(estim1, estim1_solution))

    def test_Kriging2points2D(self):
        """  TEST METHOD

        Kriging case with two points in 2D and a linear variogram.
        Reference values: Kitanidis (1996) page 170
        """
        x = np.array([[9.7, 47.6], [43.8, 24.6]])
        v = np.array([1.22, 2.822])
        xu = np.array([[18.8, 67.9]])
        vario = geo.Variogram(rang=30, sill=0.006, typ="linear", nugget=0.1)
        estim2, var2 = geo.ordinary_kriging_variogram(x, v, xu, vario)

        estim2_solution = 1.6364
        var2_solution = 0.4201


        self.assertTrue(float_eq(estim2, estim2_solution))
        self.assertTrue(float_eq(var2, var2_solution))

if __name__ == '__main__':
    unittest.main(verbosity=2)