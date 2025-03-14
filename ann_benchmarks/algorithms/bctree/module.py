import numpy
import bc_tree
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
        # Convert to float32 as required by the C++ implementation
        self._data = X.astype(numpy.float32)

        
        # add a column of ones to the data to align with the distance formula in the C++ code
        ones_column = numpy.ones((self._data.shape[0], 1), dtype=numpy.float32)
        self._data = numpy.hstack((self._data, ones_column))

        if self._metric == "angular":
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]
            
        n, d = self._data.shape
        # Ensure the array is contiguous in memory and get pointer to data
        data_array = numpy.ascontiguousarray(self._data.ravel())

        # int   n,                        // number of data points
        # int   d,                        // dimension of data points
        # int   leaf,                     // leaf size of bc-tree
        # const DType *data);             // data points
        self._tree.preprocess(n, d, self._max_leaf_size, data_array)

    def set_query_arguments(self, candidates):
        self._candidates = candidates
        # print(f"Setting candidates to {candidates}")

    def query(self, q, b, n):
        # For hyperplane queries, we need to handle the normal vector q and bias b
        qnorm = numpy.linalg.norm(q)
        # Normalize query if using angular distance
        if self._metric == "angular":
            q = q / qnorm
            b = b / qnorm
        
        q = numpy.append(q, b)
        
        # Convert query to float32 and ensure contiguous
        q = q.ravel()
		# int   top_k,                    // top_k value
        # int   cand,                     // number of candidates
        # float c,                        // approximation ratio
        # const float *query,             // input query
        # MinK_List *list);               // top-k results (return)

        # Perform the search
        results = self._tree.search(n, self._candidates, self._c, q)
        if any(idx < 0 or idx >= len(self._data) for idx in results):
            raise IndexError("Search returned out-of-bounds indices")
        return results

    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, '_data') else 0

    def __str__(self):
        return "BC_tree(leaf_size=%d, candidates=%d, c=%d)" % (self._max_leaf_size, self._candidates, self._c)