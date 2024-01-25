import numpy as np

from spread_dimension import MetricSpace

from spread_dimension.exceptions_metric_space import *

import unittest


class Tests(unittest.TestCase):

    def test_from_distance_function(self):

        bad_dist_map = {
            ('a', 'a'): 0,
            ('b', 'a'): 1,
            ('b', 'b'): 0,
            ('a', 'c'): 1,
            ('c', 'a'): 1,
            ('c', 'b'): 2,
            ('c', 'c'): 0
        }

        with self.assertRaises(InvalidDistanceFunctionError):
            ms = MetricSpace.from_distance_map(
                bad_dist_map
            )

        dist_map = {
            ('a', 'a'): 0,
            ('a', 'b'): 1,
            ('b', 'a'): 1,
            ('b', 'b'): 0,
            ('a', 'c'): 1,
            ('c', 'a'): 1,
            ('b', 'c'): 2,
            ('c', 'b'): 2,
            ('c', 'c'): 0
        }

        ms = MetricSpace.from_distance_map(dist_map)
        self.assertEqual(
            ms.distance_matrix_.shape,
            (3,3)
        )

        validated = ms.validate_distance_matrix()
        self.assertTrue(validated)

    def test_spread(self):

        dist_map = {
            ('a', 'a'): 0,
            ('a', 'b'): 1,
            ('b', 'a'): 1,
            ('b', 'b'): 0,
            ('a', 'c'): 1,
            ('c', 'a'): 1,
            ('b', 'c'): 2,
            ('c', 'b'): 2,
            ('c', 'c'): 0
        }

        ms = MetricSpace.from_distance_map(dist_map)

        spread_value = 2.520612706556268
        spread_calc = ms.spread(1)
        self.assertAlmostEqual(
            spread_calc,
            spread_value
        )

        spread_value_t_2 = 2.928043941622913
        spread_calc_2 = ms.spread(2)
        self.assertAlmostEqual(
            spread_calc_2,
            spread_value_t_2
        )

    def test_spread_dimension(self):

        dist_map = {
            ('a', 'a'): 0,
            ('a', 'b'): 1,
            ('b', 'a'): 1,
            ('b', 'b'): 0,
            ('a', 'c'): 1,
            ('c', 'a'): 1,
            ('b', 'c'): 2,
            ('c', 'b'): 2,
            ('c', 'c'): 0
        }

        ms = MetricSpace.from_distance_map(dist_map)

        spread_dim_1 = ms.spread_dimension(0.1)
        self.assertAlmostEqual(
            spread_dim_1,
            0.15769165951167904
        )

        spread_dim_2 = ms.spread_dimension(2)
        self.assertAlmostEqual(
            spread_dim_2,
            0.09656065992569603
        )

        spread_dim_3 = ms.spread_dimension(3)
        self.assertAlmostEqual(
            spread_dim_3,
            0.01979752338143859
        )

        spread_dim_9999 = ms.spread_dimension(9999)
        self.assertAlmostEqual(
            spread_dim_9999,
            0
        )

    def test_empty_distance_matrix(self):

        ms = MetricSpace()
        with self.assertRaises(DistanceMatrixNotDefined):
            ms.spread(1)

    def test_empty_distance_matrix(self):

        ms = MetricSpace()

        with self.assertRaises(DistanceMatrixNotDefined):
            ms.spread(1)

        with self.assertRaises(DistanceMatrixNotDefined):
            ms.spread_dimension(1)

        with self.assertRaises(DistanceMatrixNotDefined):
            ms._spread(1)

        with self.assertRaises(DistanceMatrixNotDefined):
            ms._spread_dimension(1)

    def test_number_of_points(self):

        dist_map = {
            ('a', 'a'): 0,
            ('a', 'b'): 1,
            ('b', 'a'): 1,
            ('b', 'b'): 0,
            ('a', 'c'): 1,
            ('c', 'a'): 1,
            ('b', 'c'): 2,
            ('c', 'b'): 2,
            ('c', 'c'): 0
        }

        ms = MetricSpace.from_distance_map(dist_map)
        self.assertEqual(ms.number_of_points, 3)

    def test_create_from_matrix(self):
        dist_mat = create_distance_matrix(50)

        ms = MetricSpace(dist_mat)
        self.assertEqual(ms.number_of_points, 50)
        self.assertEqual(ms.distance_matrix_.shape,(50,50))

    def test_create_partial_submatrix(self):
        dist_mat = DM
        ms = MetricSpace(dist_mat)
        ms.get_random_partial_submatrix(5)

        self.assertEqual(
            ms.partial_distance_matrix.shape,
            (5,10)
        )

    def test_create_partial_submatrix(self):
        dist_mat = DM
        ms = MetricSpace(dist_mat)
        with self.assertRaises(InvalidSampleError):
            ms.get_random_partial_submatrix(20)


    def test_pseudo_spread(self):
        dist_mat = DM
        ms = MetricSpace(dist_mat)
        ms.select_partial_submatrix([0,3,5,8])
        pseudo_spread = ms.pseudo_spread(2)

        pseudo_spread = ms.pseudo_spread(1)

        self.assertEqual(
            ms.partial_distance_matrix.shape,
            (4,10)
        )
        self.assertAlmostEqual(
            np.sum(pseudo_spread),
            1.042057588075496
        )


    def test_pseudo_spread_dimension(self):
        dist_mat = DM
        ms = MetricSpace(dist_mat)
        ms.select_partial_submatrix([0,3,5,8])
        pseudo_spread_dim = ms.pseudo_spread_dimension(2)
        pseudo_spread_dim = ms.pseudo_spread_dimension(1)

        self.assertEqual(
            ms.partial_distance_matrix.shape,
            (4,10)
        )
        self.assertAlmostEqual(
            np.sum(pseudo_spread_dim),
            0.8386878673024255
            )

    def test_validate_distance_matrix_triang_fail(self):

        dist_mat = DM
        ms = MetricSpace(dist_mat)

        triang = ms.satisfies_triangle_inequality()
        self.assertFalse(triang)

        validated = ms.validate_distance_matrix()
        self.assertFalse(validated)

    def test_validate_distance_matrix_not_def(self):

        ms = MetricSpace()
        with self.assertRaises(DistanceMatrixNotDefined):
            validated = ms.validate_distance_matrix()


DM = np.array(
[[0.        , 0.86023494, 0.84414229, 0.78592441, 1.89314045,
        1.45961507, 1.26634311, 1.58509588, 1.27898231, 0.78813914],
       [0.86023494, 0.        , 0.80763595, 0.59157119, 1.43804199,
        1.93908669, 0.81180936, 1.13239248, 1.21725104, 1.28826069],
       [0.84414229, 0.80763595, 0.        , 0.70331808, 1.04041169,
        1.77750298, 0.64170064, 1.29919534, 0.83832847, 0.26515326],
       [0.78592441, 0.59157119, 0.70331808, 0.        , 0.69461367,
        1.182704  , 1.31599872, 1.13780024, 0.71928061, 1.02838491],
       [1.89314045, 1.43804199, 1.04041169, 0.69461367, 0.        ,
        1.38314669, 0.96916901, 1.30487894, 1.79943345, 1.2156519 ],
       [1.45961507, 1.93908669, 1.77750298, 1.182704  , 1.38314669,
        0.        , 1.48345696, 1.46063287, 1.31303329, 1.28054466],
       [1.26634311, 0.81180936, 0.64170064, 1.31599872, 0.96916901,
        1.48345696, 0.        , 1.739612  , 1.41184998, 1.25314192],
       [1.58509588, 1.13239248, 1.29919534, 1.13780024, 1.30487894,
        1.46063287, 1.739612  , 0.        , 1.3786388 , 0.14606098],
       [1.27898231, 1.21725104, 0.83832847, 0.71928061, 1.79943345,
        1.31303329, 1.41184998, 1.3786388 , 0.        , 0.66269662],
       [0.78813914, 1.28826069, 0.26515326, 1.02838491, 1.2156519 ,
        1.28054466, 1.25314192, 0.14606098, 0.66269662, 0.        ]]
)


def create_distance_matrix(n):
    A = np.random.rand(n, n)
    B = A - np.diag(np.diag(A))
    C = B+B.T
    return C



if __name__=='__main__':
    unittest.main()
