import numpy
import fh
from ..base.module import BaseANN

class FH(BaseANN):
    def __init__(self, metric, method_param):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("FH doesn't support metric %s" % metric)
        self._metric = metric
        self._m_hashers = method_param["m_hashers"]
        self._scale_factor = method_param["scale_factor"]
        self._interval_ratio = 0.9
        self._sep_threshold = method_param["sep_threshold"]
        self._candidates = method_param["candidates"]
        self._tree = fh.FH()
        self.name = f"FH(m={self._m_hashers}, s={self._scale_factor}, b={self._interval_ratio}, sep_thresh={self._sep_threshold}, candidates={self._candidates}"

    def index(self, X):
        # Convert to float32 as required by the C++ implementation
        self._data = X.astype(numpy.float32)
        if self._metric == "angular":
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]
        
        n, d = self._data.shape
        # Ensure the array is contiguous in memory and get pointer to data
        data_array = numpy.ascontiguousarray(self._data.ravel())

        # int   n,                            // number of input data
        # int   d,                            // dimension of input data
        # int   m,                            // #hashers
        # int   s,                            // scale factor of dimension
        # float b,                            // interval ratio
        # const DType *data)                  // input data
        self._tree.preprocess(n, d, self._m_hashers, self._scale_factor, self._interval_ratio, data_array)

    def query(self, q, b, n):
        # For hyperplane queries, we need to handle the normal vector q and bias b
        # Normalize query if using angular distance
        if self._metric == "angular":
            q = q / numpy.linalg.norm(q)
        
        # Convert query to float32 and ensure contiguous
        q = numpy.ascontiguousarray(q.astype(numpy.float32))
		# int   top_k,                    // top_k value
        # int   l,                        // separation threshold, (num buckets to appear in to be a candidate)
        # int   cand,                     // number of candidates
        # const float *query,             // input query
        # MinK_List *list);               // top-k results (return)

        # Perform the search
        results = self._tree.search(n, self._sep_threshold, self._candidates, q)
        if any(idx < 0 or idx >= len(self._data) for idx in results):
            raise IndexError("Search returned out-of-bounds indices")
        return results

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, '_data') else 0
