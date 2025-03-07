import numpy
import bc_tree
from ..base.module import BaseANN

class BC_tree(BaseANN):
    def __init__(self, metric, method_param):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError("BC_tree doesn't support metric %s" % metric)
        self._metric = metric
        self._leaf_size = method_param["leaf_size"]
        self._candidates = 10 # Default candidates set to 10
        self._c = 10.0  # Ratio c set to 10.0 as default
        self._tree = bc_tree.BCTree()
        self.name = f"BC_tree(leaf_size={self._leaf_size}, candidates={self._candidates})"

    def index(self, X):
        # Convert to float32 as required by the C++ implementation
        self._data = X.astype(numpy.float32)
        if self._metric == "angular":
            self._data = self._data / numpy.linalg.norm(self._data, axis=1)[:, numpy.newaxis]
        elif self._metric == "euclidean":
            self._data = self._data
        
        n, d = self._data.shape
        # Ensure the array is contiguous in memory and get pointer to data
        data_array = numpy.ascontiguousarray(self._data.flatten())
        self._tree.preprocess(n, d, self._leaf_size, data_array)

    def query(self, q, b, n):
        # For hyperplane queries, we need to handle the normal vector q and bias b
        # Normalize query if using angular distance
        if self._metric == "angular":
            q = q / numpy.linalg.norm(q)
        
        # Convert query to float32 and ensure contiguous
        q = numpy.ascontiguousarray(q.astype(numpy.float32).flatten())
        
        # Perform the search
        results = self._tree.search(n, self._candidates, self._c, q)
        # HACK: to convert 1-based indices to 0-based - in case error appears again?
        # return [idx - 1 for idx in results]
        return results
    
    def get_memory_usage(self):
        # Return an estimate of memory usage in bytes
        # This is a rough estimate based on the data size
        return self._data.nbytes if hasattr(self, '_data') else 0
