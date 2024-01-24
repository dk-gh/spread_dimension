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
    0) d(x,y)=>0
    1a) if d(x,y)=0 then x=y
    1b) if x=y then d(x,y)=0
    2) d(x,y) = d(y,x)
    3) d(x,z) <= d(x,y) + d(y,z)

    The distance function is represented by the distance matrix,
    an nxn matrix D where n = |X|. The entry D[i,j]=d(x_i, x_j).

    Note that Axiom 1 means the diagonal is zero, and Axiom 2 implies
    the matrix is symmetric, i.e. D[i,j] = D[j,i] for all i,j.
    """

    def __init__(self, distance_matrix_=None):
        """
        Metric space is essentially defined by its distance matrix.
        We can create a metric space by specifying the distance
        matrix directly, or we can instantiate the object and create
        distance matrix separately.
        """
        self.distance_matrix_ = distance_matrix_
        self.partial_distance_matrix = None

    def validate_distance_matrix(self):

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        A = self.distance_matrix_

        axiom_0 = True
        axiom_1a = True
        axiom_1b = True
        axiom_2 = True
        axiom_3 = True

        B = A+np.diag(np.diag(A)+1)
        if np.any(B==0):
            axiom_1a = False

        if np.any(A<0):
            axiom_0 = False

        if not (A==A.T).all():
            axiom_2 = False

        if np.any(np.diag(A)>0):
            axiom_1b = False

        for row in A:
            for i, dist_i in enumerate(row[:]):
                for j, dist_j in enumerate(row[i+1:]):
                    k = j+i+1
                    check = A[i,k] <= row[i] + row[k]

                    if not check:
                        axiom_3 = False
                        break
                if not axiom_3:
                    break
            if not axiom_3:
                break

        axioms = (axiom_0, axiom_1a, axiom_1b, axiom_2, axiom_3)
        is_valid = functools.reduce(
            lambda x,y: x and y,
            axioms
        )

        return is_valid, axioms

    @property
    def number_of_points(self):
        if self.distance_matrix_ is not None:
            return self.distance_matrix_.shape[1]
        elif self.partial_distance_matrix:
            return self.partial_distance_matrix.shape[1]
        else:
            return 0

    def spread(self, t):
        """
        The spread of a metric space as defined by Simon Willerton.
        The spread here is implemented as a function on the distance
        matrix.

        This is a vectorised implementation.
        """

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        A = self.distance_matrix_
        D = ne.evaluate("exp(-t*A)")

        return np.sum(1/np.sum(D, axis=0))

    def _spread(self, t):
        """
        The spread of a metric space as defined by Simon Willerton.
        The spread here is implemented as a function on the distance
        matrix.

        This is a vectorised implementation.
        """

        if self.distance_matrix_ is None:
            raise DistanceMatrixNotDefined

        A = self.distance_matrix_
        E = np.exp(-t*A)
        return np.sum(1/np.sum(E, axis=0))

    @staticmethod
    def propagation_of_error(mean_X, varX, mean_Y, varY, covXY):
        A = (mean_X/mean_Y)**2
        B = varX/(mean_X**2)
        C = varY/(mean_Y**2)
        D = 2*covXY/(mean_X*mean_Y)
        var = A*(B+C-D)
        return var

    def pseudo_spread(self, t):

        if self.partial_distance_matrix is None:
            raise PartialDistanceMatrixNotDefined

        D = np.sum(
            np.exp(-t*self.partial_distance_matrix),
            axis=1
        )

        return 1/D

    def pseudo_spread_dimension(self, t):
        """
        Returns the spread dimension of distance
        matrix when scaled by a factor of t.

        This is a vectorised implementation of the
        exact formula for spread dimension of a finite
        metric space.
        """

        n, m = self.partial_distance_matrix.shape

        Y = self.pseudo_spread(t)
        mean_Y = np.sum(Y)/n
        varY = statistics.variance(Y, mean_Y)

        D = self.partial_distance_matrix
        E = ne.evaluate("exp(-t*D)")

        X = np.sum(t*D*E, axis=1)/(np.sum(E, axis=1)**2)
        mean_X = np.sum(X)/n
        varX = statistics.variance(X, mean_X)

        covXY = statistics.covariance(X, Y)

        G = mean_X/mean_Y

        varG = MetricSpace.propagation_of_error(
            mean_X,
            varX,
            mean_Y,
            varY,
            covXY
        )

        return G, varG

    def spread_dimension(self, t):
        """
        Returns the spread dimension of distance
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
        """
        Returns the spread dimension of distance
        matrix when scaled by a factor of t.


        Implemented using numpy only.
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
        """
        Takes a distance map in the form of a dict with entries
        d = {(x,x):0, (x,y):...} and generates an adjacency matrix
        from this data, and returns a metric space with this
        adjacency matrix.

        This is slow for large metric spaces, but useful for creating
        some spaces.
        """
        M = len(distance_map)
        N = int(np.sqrt(M))

        if N*N != M:
            raise InvalidDistanceFunctionError(
                'Invalid distance function: number of elements in the domain'
                f' ({M}) should be a square number'
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

        distance_matrix_ = distance_matrix_ + np.transpose(distance_matrix_)

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

