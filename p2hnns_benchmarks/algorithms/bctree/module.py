import numpy
import bc_tree
import psutil
from ..base.module import BaseANN


class BC_tree(BaseANN):
    def __init__(self, metric, max_leaf_size):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("BC_tree doesn't support metric %s" % metric)
        self._metric = metric
        self._max_leaf_size = max_leaf_size
        self._c = 10.0
        self._tree = bc_tree.BCTree()

    def index(self, X):
        self._data = X.astype(numpy.float32)
        
        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]
        
        # Always add ones column after any normalization
        ones_column = numpy.ones((self._data.shape[0], 1), dtype=numpy.float32)
        self._data = numpy.hstack((self._data, ones_column))
        
        n, d = self._data.shape
        data_array = numpy.ascontiguousarray(self._data.ravel())
        self._tree.preprocess(n, d, self._max_leaf_size, data_array)

    def set_query_arguments(self, candidates):
        self._candidates = candidates
        # print(f"Setting candidates to {candidates}")


    def query(self, q, b, n):
        qnorm = numpy.linalg.norm(q)

        if self._metric == "angular":
            q_normalized = q / qnorm
            b_normalized = b / qnorm
            q_to_pass = numpy.append(q_normalized, b_normalized)
        elif self._metric == "euclidean":
            q_to_pass = numpy.append(q, b)

        q_to_pass = q_to_pass.astype(numpy.float32)
        q_to_pass = numpy.ascontiguousarray(q_to_pass)

        results, self._num_lin_scans = self._tree.search(n, self._candidates, self._c, q_to_pass)
        return results
    
    def get_additional(self):
        return {"dist_comps": self._num_lin_scans}

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024

    def __str__(self):
        return "BC_tree(leaf_size=%d, candidates=%d, c=%f)" % (self._max_leaf_size, self._candidates, self._c)
