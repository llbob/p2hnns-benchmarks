import numpy
import mh
from ..base.module import BaseANN


class MH(BaseANN):
    def __init__(self, metric, M_proj_vectors, m_single_hashers, l_hash_tables):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("MH doesn't support metric %s" % metric)
        self._metric = metric
        self._M_proj_vectors = M_proj_vectors
        self._m_single_hashers = m_single_hashers
        self._l_hash_tables = l_hash_tables
        self._interval_ratio = 0.9
        self._mh_index = mh.MH()

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
        self._mh_index.preprocess(
            n, d, self._M_proj_vectors, self._m_single_hashers, self._l_hash_tables, self._interval_ratio, data_array
        )

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

        results, self._num_lin_scans = self._mh_index.search(n, self._candidates, q_to_pass)
        return results
    
    def get_additional(self):
        return {"dist_comps": self._num_lin_scans}

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, "_data") else 0

    def __str__(self):
        return "MH(M_proj_vectors=%d, m_single_hashers=%d, l_hash_tables=%d, interval_ratio=%f, candidates=%d)" % (
            self._M_proj_vectors,
            self._m_single_hashers,
            self._l_hash_tables,
            self._interval_ratio,
            self._candidates,
        )
