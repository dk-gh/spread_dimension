import numpy as np
from spread_dimension import EuclideanSubspace

from spread_dimension.exceptions_euclidean_subspace import *

import unittest


def swiss_roll():
    np.random.seed(101)
    """
    Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
    by Stephen Marsland (http://stephenmonika.net)

    You are free to use, change, or redistribute the code in any way you wish for
    non-commercial purposes, but please maintain the name of the original author.
    This code comes with no warranty of any kind.

    Stephen Marsland, 2008, 2014
    """

    n=1000

    t = 3*np.pi/2 * (1 + 2*np.random.rand(1, n))
    h = 21*np.random.rand(1, n)

    data = np.concatenate((t*np.cos(t), h, t*np.sin(t)))

    return data.T


class EuclideanSubspaceTests(unittest.TestCase):

    def test_swiss_roll_numexpr_spread_dimension(self):
        points = swiss_roll()

        sr = EuclideanSubspace(points)

        sr.compute_metric()

        spread_dimension = sr.spread_dimension(0.5)
        self.assertAlmostEqual(
            spread_dimension,
            2.0255147320496447
        )

        spread_dimension = sr.spread_dimension(1)
        self.assertAlmostEqual(
            spread_dimension,
            1.5122389544408892
        )

        spread_dimension = sr.spread_dimension(5)
        self.assertAlmostEqual(
            spread_dimension,
            0.2065511059057061
        )

    def test_swiss_roll_numpy_spread_dimension(self):
        points = swiss_roll()

        sr = EuclideanSubspace(points)

        sr.compute_metric()

        t = 0.1
        spread_dimension = sr.spread_dimension(t)
        _spread_dimension = sr._spread_dimension(t)
        self.assertAlmostEqual(
            spread_dimension,
            _spread_dimension
        )

        t = 0.5
        spread_dimension = sr.spread_dimension(t)
        _spread_dimension = sr._spread_dimension(t)
        self.assertAlmostEqual(
            spread_dimension,
            _spread_dimension
        )


        t = 5
        spread_dimension = sr.spread_dimension(t)
        _spread_dimension = sr._spread_dimension(t)
        self.assertAlmostEqual(
            spread_dimension,
            _spread_dimension
        )


        t = 50
        spread_dimension = sr.spread_dimension(t)
        _spread_dimension = sr._spread_dimension(t)
        self.assertAlmostEqual(
            spread_dimension,
            _spread_dimension
        )

    def test_empty_distance_matrix(self):

        ms = EuclideanSubspace([(1,2),(3,4)])

        with self.assertRaises(DistanceMatrixNotDefined):
            ms.spread(1)

        with self.assertRaises(DistanceMatrixNotDefined):
            ms.spread_dimension(1)

        with self.assertRaises(DistanceMatrixNotDefined):
            ms._spread(1)

        with self.assertRaises(DistanceMatrixNotDefined):
            ms._spread_dimension(1)

    def test_number_of_points(self):

        points = swiss_roll()

        sr = EuclideanSubspace(points)
        n = sr.number_of_points

        self.assertEqual(n, 1000)

        sr.compute_metric()

        self.assertEqual(
            (1000,1000),
            sr.distance_matrix_.shape
        )

    def test_random_sample_too_big(self):
        points = swiss_roll()

        sr = EuclideanSubspace(points)

        with self.assertRaises(InvalidSampleError):
            sr.compute_random_partial_matrix(1005)

    def test_random_sample(self):

        points = swiss_roll()

        sr = EuclideanSubspace(points)

        sr.compute_random_partial_matrix(10)

        n, m = sr.partial_distance_matrix.shape

        self.assertEqual((n,m), (10,1000))

    def test_valid_matrix(self):

        points = swiss_roll()

        sr = EuclideanSubspace(points)

        sr.compute_metric()

        self.assertTrue(sr.validate_distance_matrix())


if __name__=='__main__':
    unittest.main()

