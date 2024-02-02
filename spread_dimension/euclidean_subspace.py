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
        """This property overwrites the method inherited from
        the MetricSpace class which is based on dimensions
        of distance_matrix_ and/or partial_distance_matrix only
        """

        return len(self.points)

    def compute_metric(self, p_norm=2):
        """Computes the pairwise distances of all points using
        the p-norm distance d(x,y) = ||x-y||p where for
        x = (x1,x2,...xn)

        ||x||p = (|x1|^p + |x2|^p +...+|xn|^p)^(1/p)

        default value is p=2, which is the standard Euclidean
        distance.
        """

        D = scipy.spatial.distance.pdist(
                self.points,
                metric='minkowski',
                p=p_norm
            )

        self.distance_matrix_ = scipy.spatial.distance.squareform(D)

    def compute_partial_metric(self, list_of_indices, p_norm=2):
        """Computes the pairwise distances between a chosen subset
        specified by the list of indices and the whole set using
        the p-norm distance d(x,y) = ||x-y||p where for
        x = (x1,x2,...xn)

        ||x||p = (|x1|^p + |x2|^p +...+|xn|^p)^(1/p)

        The default value is p=2 which is the standard Euclidean
        distance.
        """

        chosen_points = np.take(self.points, list_of_indices, axis=0)

        self.partial_distance_matrix = scipy.spatial.distance.cdist(
                chosen_points,self.points,
                metric='minkowski',
                p=p_norm
            )

    def compute_random_partial_matrix(self, n, p_norm=2):
        """Computes the partial distance matrix for a random
        sample of n points.
        """
        N = self.number_of_points

        if n > N:
            raise InvalidSampleError(
                    f'cannot sample {n} points from {N} points'
                    )

        random_indices = random.sample(range(N), n)
        self.compute_partial_metric(
            random_indices,
            p_norm=p_norm
        )

    def find_operative_range(self, initial_scale=1, sample_size=10):
        """The range of scale values over which the spread dimension
        of a set of points in meaningful varies depending on the
        nature of the space.

        Eventually, for large enough scale values t, the (pseudo)
        spread of the space tends towards the number of points in
        the sapce.

        This method iteratively computes the pseudo spread for a
        sample of size sample_size, increasing or decreasing the
        scale value until a value between 95% and 97% of the number
        of points in the space is identified.

        The spread dimension is defined in terms of the growth of
        the spread, and hence we can identify the range over which
        the growth is maximised.

        This method is to be used as follows

            T = self.find_operative_range()

            for t in np.linspace(0,T,100):
                self.spread_dimension(t)

        The spread dimension beyond values of T will typically
        tend towards zero.
        """
        if self.distance_matrix_ is not None:
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

        return t

