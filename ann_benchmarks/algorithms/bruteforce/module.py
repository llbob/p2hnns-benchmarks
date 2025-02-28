import numpy

from ...distance import compute_distance, metrics as pd
from ..base.module import BaseANN

class BruteForce(BaseANN):
    def __init__(self, metric):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.name = "BruteForce()"

    def index(self, X):
        if self._metric == "euclidean":
            self._data = X
        elif self._metric == "angular":
            self._data = X/numpy.linalg.norm(X, axis=-1, keepdims=True)


    def query(self, q, b, n):
        return [index for index, _ in self.query_with_distances(q, b, n)]


    def query_with_distances(self, q, b, n):
        """Find indices and distances of `n` most similar vectors from the index to query
        hyperplane with normal vector `q` and bias `b`."""
        qnorm = numpy.linalg.norm(q)
        distances = numpy.abs(numpy.dot(self._data, q) + b)/qnorm
        n_smallest = numpy.argpartition(distances, n)[:n]
        # Return (index, distance) pairs
        return [(idx, distances[idx]) for idx in n_smallest]



class BruteForceBLAS(BaseANN):
    """kNN search to a hyperplane that uses a linear scan = brute force."""

    def __init__(self, metric, precision=numpy.float32):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("BruteForceBLAS doesn't support metric %s" % metric)
        self._metric = metric
        self._precision = precision
        self.name = "BruteForceBLAS()"

    def index(self, X):
        """Initialize the search index."""
        if self._metric == "angular":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            # normalize index vectors to unit length
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == "euclidean":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"

    def query(self, q, b, n):
        return [index for index, _ in self.query_with_distances(q, b, n)]

    def query_with_distances(self, q, b, n):
        """Find indices of `n` most similar vectors from the index to query
        hyperplane with normal vector `q` and bias `b`."""

        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        if self._metric == "angular" or self._metric == "euclidean":
            # use the same distance function for both metrics
            qnorm = numpy.linalg.norm(q)
            dists = numpy.abs(numpy.dot(self.index, q) + b)/qnorm
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"
        
        nearest_indices = numpy.argpartition(dists, n)[:n]
        indices = [idx for idx in nearest_indices if pd[self._metric].distance_valid(dists[idx])]

        def fix(index):
            ep = self.index[index]
            ev = q
            return (index, pd[self._metric].distance(ep, ev))

        return map(fix, indices)
