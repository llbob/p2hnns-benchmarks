import numpy
import b_tree
import mh
from ..base.module import BaseANN


class BT_MQH(BaseANN):
    def __init__(self, metric, max_leaf_size):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("MQH doesn't support metric %s" % metric)
        # mqh specific (currently mqh uses linear scan as placeholder)
        self._metric = metric

        # bt specific
        self._max_leaf_size = max_leaf_size
        self._c = 10.0
        self._tree_candidates = b_tree.BTree()

    def index(self, X):
        self._data = X.astype(numpy.float32)

        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]

        # Always add ones column after any normalization
        ones_column = numpy.ones((self._data.shape[0], 1), dtype=numpy.float32)
        self._candidate_data = numpy.hstack((self._data, ones_column))

        n, d = self._candidate_data.shape
        data_array = numpy.ascontiguousarray(self._candidate_data.ravel())
        self._tree_candidates.preprocess(n, d, self._max_leaf_size, data_array)

    def set_query_arguments(self, candidates, inital_topk):
        self._candidates = candidates
        self._initial_topk = inital_topk
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

        candidate_indices = self._tree_candidates.search(self._initial_topk, self._candidates, self._c, q_to_pass)

        candidate_indices = numpy.array(candidate_indices, dtype=numpy.int64)
    
        candidate_vectors = self._data[candidate_indices]

        if self._metric == "angular":
            distances = numpy.abs(numpy.dot(candidate_vectors, q_normalized) + b_normalized)
        elif self._metric == "euclidean":
            distances = numpy.abs(numpy.dot(candidate_vectors, q) + b) / qnorm

        nearest_indices = numpy.argpartition(distances, n)[:n]

        return list(candidate_indices[nearest_indices]) # we need to map back to the original indices

    def get_additional(self):
        return {"dist_comps": self._candidates}

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, "_data") else 0

    def __str__(self):
        return "BT_MQH(max_leaf_size=%d, candidates=%d)" % (self._max_leaf_size, self._candidates)


class MH_MQH(BaseANN):
    def __init__(self, metric, M_proj_vectors, m_single_hashers, l_hash_tables):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("MQH doesn't support metric %s" % metric)
        # mqh specific (currently mqh uses linear scan as placeholder)
        self._metric = metric

        # mh specific
        self._M_proj_vectors = M_proj_vectors
        self._m_single_hashers = m_single_hashers
        self._l_hash_tables = l_hash_tables
        self._interval_ratio = 0.9
        self._mh_index_candidates = mh.MH()

    def index(self, X):
        self._data = X.astype(numpy.float32)

        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]

        # Always add ones column after any normalization
        ones_column = numpy.ones((self._data.shape[0], 1), dtype=numpy.float32)
        self._candidate_data = numpy.hstack((self._data, ones_column))

        n, d = self._candidate_data.shape
        data_array = numpy.ascontiguousarray(self._candidate_data.ravel())

        self._mh_index_candidates.preprocess(
            n, d, self._M_proj_vectors, self._m_single_hashers, self._l_hash_tables, self._interval_ratio, data_array
        )

    def set_query_arguments(self, candidates, initial_topk):
        self._candidates = candidates
        self._initial_topk = initial_topk

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

        candidate_indices = self._mh_index_candidates.search(self._initial_topk, self._candidates, q_to_pass)

        candidate_indices = numpy.array(candidate_indices, dtype=numpy.int64)
    
        candidate_vectors = self._data[candidate_indices]

        if self._metric == "angular":
            distances = numpy.abs(numpy.dot(candidate_vectors, q_normalized) + b_normalized)
        elif self._metric == "euclidean":
            distances = numpy.abs(numpy.dot(candidate_vectors, q) + b) / qnorm

        nearest_indices = numpy.argpartition(distances, n)[:n]
        return list(candidate_indices[nearest_indices])

    def get_additional(self):
        return {"dist_comps": self._candidates}

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, "_data") else 0

    def __str__(self):
        return "MQH(max_leaf_size=%d, candidates=%d)" % (self._max_leaf_size, self._candidates)


class MQH(BaseANN):
    def __init__(self, metric):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("MQH doesn't support metric %s" % metric)
        # mqh specific (currently mqh uses linear scan as placeholder)
        self._metric = metric

    def index(self, X):
        self._data = X.astype(numpy.float32)

        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]

    def set_query_arguments(self, candidates):
        self._candidates = candidates
        # print(f"Setting candidates to {candidates}")

    def query(self, q, b, n):

        qnorm = numpy.linalg.norm(q)

        if self._metric == "angular":
            q_normalized = q / qnorm
            b_adjusted = b / qnorm
            distances = numpy.abs(numpy.dot(self._data, q_normalized) + b_adjusted)
        elif self._metric == "euclidean":
            distances = numpy.abs(numpy.dot(self._data, q) + b) / qnorm
        else:
            assert False, "invalid metric"

        nearest_indices = numpy.argpartition(distances, n)[:n]
        return list(nearest_indices)

    def get_additional(self):
        return {"dist_comps": self._candidates}

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, "_data") else 0

    def __str__(self):
        return "MQH(max_leaf_size=%d, candidates=%d)" % (self._max_leaf_size, self._candidates)
