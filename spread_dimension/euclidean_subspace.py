import numpy as np
import random
import scipy.spatial

from .metric_space import MetricSpace

from .exceptions_euclidean_subspace import *

class EuclideanSubspace(MetricSpace):
    """
    A Euclidean subspace is a subset of points of R^n
    for some n.

    These are useful examples of metric spaces, with
    a variety of possible metrics.

    It will also be useful to keep track of the underlying
    points because for example we may want to implement
    noise reduction techniques on these subspaces.

    We store these points as a list of tuples, where
    the tuple represents the coordinates in n-dimensional
    Euclidean space
    """

    def __init__(self, list_of_points):
        self.points = np.array(list_of_points)
        MetricSpace.__init__(self)

    @property
    def number_of_points(self):
        return len(self.points)

    def compute_metric(self, p_norm=2):
        """
        computes the pairwise distances of all points
        using the p-norm distance d(x,y) = ||x-y||p where
        for x = (x1,x2,...xn)

        ||x||p = (|x1|^p + |x2|^p +...+|xn|^p)^(1/p)

        default value is p=2 which is the standard Euclidean
        distance.

        uses scipy implementation of distance_matrix for speed
        """

        D = scipy.spatial.distance.pdist(
                self.points,
                metric='minkowski',
                p=p_norm
            )

        self.distance_matrix_ = scipy.spatial.distance.squareform(D)

    def compute_partial_metric(
            self,
            list_of_indices,
            p_norm=2,
            ):

        """
        computes the pairwise distances between a chosen subset of points
        and the whole set using the p-norm distance d(x,y) = ||x-y||p where
        for x = (x1,x2,...xn)

        ||x||p = (|x1|^p + |x2|^p +...+|xn|^p)^(1/p)

        default value is p=2 which is the standard Euclidean
        distance.

        uses scipy implementation of distance_matrix for speed
        """

        chosen_points = np.take(self.points, list_of_indices, axis=0)

        self.partial_distance_matrix = scipy.spatial.distance.cdist(
                chosen_points,self.points,
                metric='minkowski',
                p=p_norm
            )

    def compute_random_partial_matrix(self, n, p_norm=2):

        N = self.number_of_points

        if n > N:
            raise InvalidSampleError(
                    f'cannot sample {n} points from a space of {N} points'
                    )

        random_indices = random.sample(range(N), n)
        self.compute_partial_metric(
            random_indices,
            p_norm=p_norm
        )

    def find_operative_range(self, initial_scale=1, sample_size=10):
        
        if self.distance_matrix_:
            self.get_random_partial_submatrix(sample_size)

        else:
            self.compute_random_partial_matrix(sample_size)

        N = self.number_of_points

        lower_interval_bound = 0.95*N
        upper_interval_bound = 0.97*N

        if lower_interval_bound==upper_interval_bound:
            lower_interval_bound = N-1
            upper_interval_bound = N

        track_scales = []
        t = initial_scale
        prev_t = 0

        while True:

            track_scales.append(t)
            current_spread = self.pseudo_spread(t)

            if prev_t < t:
                if current_spread < lower_interval_bound:
                    prev_t = t
                    t = 2*t
                elif current_spread >= upper_interval_bound:
                    tmp = t
                    t = (t+prev_t)/2
                    prev_t = tmp
                else:
                    break

            elif prev_t > t:
                if current_spread <= lower_interval_bound:
                    tmp = t
                    t = (t+prev_t)/2
                    prev_t = tmp
                elif current_spread > upper_interval_bound:
                    prev_t = t
                    t = t/2
                else:
                    break

        return t, track_scales

