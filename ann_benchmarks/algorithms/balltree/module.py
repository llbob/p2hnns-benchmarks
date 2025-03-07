import numpy
import b_tree
from ..base.module import BaseANN

class B_tree(BaseANN):
    def __init__(self, metric, method_param):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("B_tree doesn't support metric %s" % metric)
        self._metric = metric
        self._max_leaf_size = method_param["max_leaf_size"]
        self._candidates = method_param["candidates"]
        self._c = 10.0
        self._tree = b_tree.BTree()
        self.name = f"B_tree(leaf_size={self._max_leaf_size}, candidates={self._candidates}, c={self._c})"

    def index(self, X):
        # Convert to float32 as required by the C++ implementation
        self._data = X.astype(numpy.float32)
        if self._metric == "angular":
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]
        
        n, d = self._data.shape
        # Ensure the array is contiguous in memory and get pointer to data
        data_array = numpy.ascontiguousarray(self._data.flatten())

        # int   n,                        // number of data points
        # int   d,                        // dimension of data points
        # int   leaf,                     // leaf size of b-tree
        # const DType *data);             // data points
        self._tree.preprocess(n, d, self._max_leaf_size, data_array)

    def query(self, q, b, n):
        # For hyperplane queries, we need to handle the normal vector q and bias b
        # Normalize query if using angular distance
        if self._metric == "angular":
            q = q / numpy.linalg.norm(q)
        
        # Convert query to float32 and ensure contiguous
        q = numpy.ascontiguousarray(q.astype(numpy.float32).flatten())
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
