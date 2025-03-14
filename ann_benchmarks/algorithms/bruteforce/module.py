import numpy

from ...distance import metrics as pd
from ..base.module import BaseANN

class BruteForce(BaseANN):
    def __init__(self, metric):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.name = "BruteForce()"

    def index(self, X):
        if self._metric == "euclidean":
            self._data = X
        elif self._metric == "angular":
            self._data = X/numpy.linalg.norm(X, axis=-1, keepdims=True)
        else:
            assert False, "invalid metric"

    def query(self, q, b, n):
        return [index for index, _ in self.query_with_distances(q, b, n)]

    def query_with_distances(self, q, b, n):
        """Find indices and distances of `n` most similar vectors from the index to query
        hyperplane with normal vector `q` and bias `b`."""
        qnorm = numpy.linalg.norm(q)
        
        if self._metric == "angular":
            q_normalized = q / qnorm
            b_adjusted = b / qnorm
            distances = numpy.abs(numpy.dot(self._data, q_normalized) + b_adjusted)
        elif self._metric == "euclidean":
            distances = numpy.abs(numpy.dot(self._data, q) + b)/qnorm
        else:
            assert False, "invalid metric"
        
        nearest_indices = numpy.argpartition(distances, n)[:n]
        return [(idx, distances[idx]) for idx in nearest_indices]

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
            assert False, "invalid metric"

    def query(self, q, b, n):
        return [index for index, _ in self.query_with_distances(q, b, n)]

    def query_with_distances(self, q, b, n):
        """Find indices and distances of `n` most similar vectors from the index to query
        hyperplane with normal vector `q` and bias `b`."""
        
        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        qnorm = numpy.linalg.norm(q)
        if self._metric == "angular":
            q_normalized = q / qnorm
            b_adjusted = b / qnorm
            distances = numpy.abs(numpy.dot(self.index, q_normalized) + b_adjusted)
        elif self._metric == "euclidean":
            distances = numpy.abs(numpy.dot(self.index, q) + b)/qnorm
        else:
            assert False, "invalid metric"
        
        nearest_indices = numpy.argpartition(distances, n)[:n]
        return [(idx, distances[idx]) for idx in nearest_indices]
