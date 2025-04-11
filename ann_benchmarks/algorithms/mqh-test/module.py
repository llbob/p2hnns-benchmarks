import numpy
import pymqh as mqh
from ..base.module import BaseANN


class MQH_test(BaseANN):
    def __init__(self, metric, M2=16, level=4, m_level=1, m_num=64):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("MQH doesn't support metric %s" % metric)
        self._metric = metric
        self._M2 = M2
        self._level = level
        self._m_level = m_level
        self._m_num = m_num

    def index(self, X):
        self._data = X.astype(numpy.float32)
        
        if self._metric == "angular":
            # Normalize for angular distance
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]
        
        n, d = self._data.shape
        self._mqh = mqh.MQH(d, self._M2, self._level, self._m_level, self._m_num)
        self._mqh.build_index(self._data)

    def set_query_arguments(self, l0, delta, flag):
        self._l0 = l0
        self._delta = delta
        self._flag = flag

    def query(self, q, b, n):
        """
        Query the index for the n nearest neighbors to the hyperplane defined by (q, b).
        
        Parameters:
        -----------
        q : numpy array
            Normal vector of the hyperplane
        b : float
            Bias term of the hyperplane equation <q, x> + b = 0
        n : int
            Number of nearest neighbors to return
            
        Returns:
        --------
        indices : numpy array
            Indices of the n nearest neighbors to the hyperplane
        """

        # Convert inputs to float32 for C++ implementation
        q = q.astype(numpy.float32)
        
        # Normalize query vector for angular distance
        if self._metric == "angular":
            qnorm = numpy.linalg.norm(q)
            if qnorm > 0:
                q = q / qnorm
                b = b / qnorm
        
        # Call the search method with the appropriate parameters
        indices, _ = self._mqh.search(q, n, b, self._l0, self._delta, self._flag)
        indices = set(indices)
        indices = list(indices)
        return indices

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        return self._data.nbytes if hasattr(self, "_data") else 0

    def __str__(self):
        return f"MQH(M2={self._M2}, level={self._level}, m_level={self._m_level}, m_num={self._m_num}, l0={self._l0}, delta={self._delta}, flag={self._flag})"