import functools
import numexpr as ne
import numpy as np
import random
import statistics

from .exceptions_metric_space import *

class MetricSpace():
    """
    A metric space consists of a set X with a function
    d: XxX -> [0, infty)
    satisfying the following axioms for all x,y,z in X:

    0) d(x,y)>=0 (strictly non-negative)
    1) d(x,y)=0 iff x=y (identity)
    2) d(x,y) = d(y,x) (symmetry)
    3) d(x,z) <= d(x,y) + d(y,z) (triangle inequality)

    The distance function is represented by the distance matrix,
    an nxn matrix D where n = |X|. The entry D[i,j]=d(x_i, x_j).

    Metric space is essentially defined by its distance matrix.
    We can create a metric space by specifying the distance
    matrix directly, or we can instantiate the object and create
    distance matrix separately.
    """

    def __init__(self, distance_matrix_=None):
        self.distance_matrix_ = distance_matrix_
        self.partial_distance_matrix = None

    def satisfies_nonnegativity(self):
        """Returns True if Axiom 0 is satisfied:
        that d(x,y)>=0 (non-negativity)
        else returns False
        """
        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        if np.any(self.distance_matrix_ < 0):
            return False
        else:
            return True

    def satisfies_identity(self):
        """Returns True if Axiom 1 is satisfied:
        that x = y iff d(x,y)==0 (identity)
        else returns False
        """
        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        N = self.number_of_points

        if np.any(np.diag(self.distance_matrix_) != 0):
            return False

        if np.any((self.distance_matrix_ + np.diag([1]*N))==0):
            return False

        return True

    def satisfies_symmetry(self):
        """Returns True if Axiom 2 is satisfied:
        that d(x,y)==d(y,x) (symmetry)
        else returns False
        """
        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        D = self.distance_matrix_

        if np.all(D==D.T):
            return True

        else:
            return False

    def satisfies_triangle_inequality(self):
        """Returns True if Axiom 3 is satisfied:
        d(x,z) <= d(x,y) + d(y,z) (triangle inequality)
        else returns False
        """
        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        rotated = self.distance_matrix_.copy()

        N = self.number_of_points

        for i in range(N-1):

            rotated = np.concatenate(
                (rotated[1:,:], rotated[[0]]),
                axis=0
            )

            point_distances = np.diag(rotated)
            min_distance_sum = np.min(
                rotated+self.distance_matrix_,
                axis=1
            )

            if np.any(point_distances > min_distance_sum):
                return False

        return True

    def validate_distance_matrix(self):
        """Returns True if Axioms 1,2,3 & 4 are
        satisfied, else returns False
        """
        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        n, m = self.distance_matrix_.shape
        if n != m:
            return False

        if not self.satisfies_nonnegativity():
            return False

        if not self.satisfies_identity():
            return False

        if not  self.satisfies_symmetry():
            return False

        if not self.satisfies_triangle_inequality():
            return False

        return True

    @property
    def number_of_points(self):
        """Returns the number of points in the metric space
        """

        if self.distance_matrix_ is not None:
            return self.distance_matrix_.shape[1]

        elif self.partial_distance_matrix:
            return self.partial_distance_matrix.shape[1]

        else:
            return 0

    def spread(self, t):
        """The spread of the metric space at scale t

        This implementation uses NumExpr for memory optimisation.
        """

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        A = self.distance_matrix_
        D = ne.evaluate("exp(-t*A)")

        return np.sum(1/np.sum(D, axis=0))

    def _spread(self, t):
        """The spread of a metric space at scale t
        implemented using numpy array operations only.

        This implementation is mostly used for testing correctness
        during optimisation.
        """

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        A = self.distance_matrix_
        E = np.exp(-t*A)
        return np.sum(1/np.sum(E, axis=0))

    @staticmethod
    def _propagation_of_error(mean_X, varX, mean_Y, varY, covXY):
        """Calculates the propagation of error for the pseudo
        spread dimension.
        """

        if mean_X*mean_Y==0:
            return 0

        A = (mean_X/mean_Y)**2
        B = varX/(mean_X**2)
        C = varY/(mean_Y**2)
        D = 2*covXY/(mean_X*mean_Y)
        var = A*(B+C-D)

        return var

    def _pseudo_spread_part_eval(self, t):

        if self.partial_distance_matrix is None:
            raise PartialDistanceMatrixNotDefined

        D = np.sum(
            np.exp(-t*self.partial_distance_matrix),
            axis=1
        )

        return 1/D

    def pseudo_spread(self, t):
        part_eval = self._pseudo_spread_part_eval(t)
        N = self.number_of_points
        return np.mean(part_eval)*N

    def pseudo_spread_dimension(self, t):
        """
        Returns the spread dimension of distance
        matrix when scaled by a factor of t.

        This is a vectorised implementation of the
        exact formula for spread dimension of a finite
        metric space.
        """

        if self.partial_distance_matrix is None:
            raise PartialDistanceMatrixNotDefined

        n, m = self.partial_distance_matrix.shape

        Y = self._pseudo_spread_part_eval(t)
        mean_Y = np.sum(Y)/n
        varY = statistics.variance(Y, mean_Y)

        D = self.partial_distance_matrix
        E = ne.evaluate("exp(-t*D)")

        X = np.sum(t*D*E, axis=1)/(np.sum(E, axis=1)**2)
        mean_X = np.sum(X)/n
        varX = statistics.variance(X, mean_X)

        covXY = statistics.covariance(X, Y)

        G = mean_X/mean_Y

        varG = MetricSpace._propagation_of_error(
            mean_X,
            varX,
            mean_Y,
            varY,
            covXY
        )

        return G, varG

    def spread_dimension(self, t):
        """Returns the spread dimension of distance
        matrix when scaled by a factor of t.

        This is a vectorised implementation of the
        exact formula for spread dimension of a finite
        metric space.

        Optimised using NumExpr.
        """

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        D = self.distance_matrix_
        E = ne.evaluate("exp(-t*D)")

        lead_factor = (t/self.spread(t))

        denomenator = np.sum(E, axis=0)**2
        ne.evaluate("D*E", out=E)
        return lead_factor * np.sum(np.sum(E, axis=0)/denomenator)

    def _spread_dimension(self, t):
        """Returns the spread dimension of distance
        matrix when scaled by a factor of t.


        Implemented using numpy array operations only. This
        is primarily used for testing correctness when optimising.
        """

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        D = self.distance_matrix_
        E = np.exp(-t*D)
        A = t/self._spread(t)
        B = np.sum((np.sum(D*E, axis=0))/(np.sum(E, axis=0)**2))
        return A*B

    @classmethod
    def from_distance_map(cls, distance_map):
        """Takes a distance map in the form of a dict with entries
        d = {(x,x):0, (x,y):...} and generates an adjacency matrix
        from this data, and returns a metric space with this
        adjacency matrix.

        This is slow for large metric spaces, but useful for
        creating toy model spaces.
        """
        M = len(distance_map)
        N = int(np.sqrt(M))

        if N*N != M:
            raise InvalidDistanceFunctionError(
                'Invalid distance function: number of elements in'
                f'the domain {M} should be a square number'
            )

        distance_matrix_ = np.zeros(N*N).reshape(N, N)

        elements = set()
        for i in distance_map:
            elements = elements.union(set(i))

        elements = list(elements)

        for i, element_i in enumerate(elements):
            for j, element_j in enumerate(elements):

                distance_matrix_[i, j] = distance_map[
                    (element_i, element_j)
                ]

        distance_matrix_ += np.transpose(distance_matrix_)

        return cls(distance_matrix_)

    def select_partial_submatrix(self, list_of_indices):

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        n = len(list_of_indices)
        N = self.number_of_points

        if n > N:
            raise InvalidSampleError(
                f'cannot sample {n} points from space with {N} points'
                )

        P = self.distance_matrix_[list_of_indices, :]
        self.partial_distance_matrix = P

    def get_random_partial_submatrix(self, n):

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        N = self.number_of_points
        if n > N:
            raise InvalidSampleError(
                f'cannot sample {n} points from space with {N} points'
                )

        sample_indices = random.sample(range(N),n)
        self.select_partial_submatrix(sample_indices)

