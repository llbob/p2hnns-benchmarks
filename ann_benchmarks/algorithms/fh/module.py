import numpy
import fh
from ..base.module import BaseANN

class FH(BaseANN):
    def __init__(self, metric, m_hashers, scale_factor, sep_threshold):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("FH doesn't support metric %s" % metric)
        self._metric = metric
        self._m_hashers = m_hashers
        self._scale_factor = scale_factor
        self._sep_threshold = sep_threshold
        self._interval_ratio = 0.9
        self._fh_index = fh.FH()

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
        self._fh_index.preprocess(n, d, self._m_hashers, self._scale_factor, self._interval_ratio, data_array)

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

        results = self._fh_index.search(n, self._sep_threshold, self._candidates, q_to_pass)
        return results

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, '_data') else 0

    def __str__(self):
        return "FH(m_hashers=%d, scale_factor=%d, sep_threshold=%d, candidates=%d)" % (self._m_hashers, self._scale_factor, self._sep_threshold, self._candidates)