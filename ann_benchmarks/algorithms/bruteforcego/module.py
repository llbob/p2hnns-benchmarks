import ctypes
import os
import numpy as np
from pathlib import Path
from ann_benchmarks.algorithms.base.module import BaseANN

class BruteForceGo(BaseANN):
    
    def __init__(self, metric, **kwargs):
        self.metric = metric
        
        # Load the shared library
        lib_path = Path(os.path.dirname(__file__)) / "binding" / "libgowrapper.so"
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Define function signatures - Note the capitalized function names
        self.lib.Index.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # data pointer
            ctypes.c_int,                    # n (number of points)
            ctypes.c_int                     # d (dimension)
        ]
        self.lib.Index.restype = None  # void return
        
        self.lib.Query.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # normal vector
            ctypes.c_float,                  # bias
            ctypes.c_int,                    # k (number of results)
            ctypes.POINTER(ctypes.c_int)     # results array
        ]
        self.lib.Query.restype = ctypes.POINTER(ctypes.c_int)  # returns pointer to results
    
    def index(self, X):
        """Build the index for the algorithm."""
        n, d = X.shape
        
        # Convert data to float32 (required by Go)
        data = X.astype(np.float32)
        
        # Get pointer to the data
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the Go Index function
        self.lib.Index(data_ptr, n, d)
    
    def query(self, q, b, k):
        """Query the index for the nearest neighbors."""
        # Convert query to float32
        q_data = q.astype(np.float32)
        
        # Get pointer to query data
        q_ptr = q_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Prepare results array
        results = np.zeros(k, dtype=np.int32)
        results_ptr = results.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        # Call the Go Query function
        self.lib.Query(q_ptr, ctypes.c_float(b), ctypes.c_int(k), results_ptr)
        
        # Return results as numpy array
        return results