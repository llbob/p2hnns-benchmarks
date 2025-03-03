import os
from base.module import BaseANN
import subprocess
import numpy as np
import tempfile
from typing import List, Any, Optional



class BallTreeWrapper:
    def __init__(self, metric: str, leaf_size: int = 1000):
        """
        Initialize the Ball-Tree algorithm.
        
        Parameters:
        -----------
        metric : str
            Distance metric ('angular' or 'euclidean')
        leaf_size : int
            Leaf size for the Ball-Tree (default: 1000)
        """
        self.metric = metric
        self.leaf_size = leaf_size
        self.binary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "external_repo/methods/p2h_search")
        self.data_path = None
        self.query_path = None
        self.results_path = None
        self.n = 0
        self.d = 0
        
    def fit(self, X: np.ndarray) -> None:
        """
        Build the Ball-Tree index from the data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data points to index
        """
        self.n, self.d = X.shape
        
        # Create temporary files for data
        fd, self.data_path = tempfile.mkstemp(suffix='.ds')
        os.close(fd)
        
        # Write data to binary file
        X = X.astype(np.float32)
        X.tofile(self.data_path)
        
        # Create directory for results
        fd, self.results_path = tempfile.mkstemp(suffix='.out')
        os.close(fd)
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
    def query(self, X: np.ndarray, k: int) -> List[List[int]]:
        """
        Query the Ball-Tree index.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Query points
        k : int
            Number of nearest neighbors to return
            
        Returns:
        --------
        List[List[int]]
            Indices of the k nearest neighbors for each query point
        """
        qn, d = X.shape
        assert d == self.d, f"Query dimension {d} doesn't match data dimension {self.d}"
        
        # Create temporary file for queries
        fd, self.query_path = tempfile.mkstemp(suffix='.q')
        os.close(fd)
        
        # Write queries to binary file
        X = X.astype(np.float32)
        X.tofile(self.query_path)
        
        # Create temporary file for ground truth (not used but required by the binary)
        fd, truth_path = tempfile.mkstemp(suffix='.gt')
        os.close(fd)
        
        # Run the binary
        cmd = [
            self.binary_path,
            "-alg", "2",  # Ball-Tree algorithm
            "-n", str(self.n),
            "-qn", str(qn),
            "-d", str(self.d),
            "-leaf", str(self.leaf_size),
            "-cf", "config",
            "-dt", "float32",
            "-dn", "Dataset",
            "-ds", self.data_path,
            "-qs", self.query_path,
            "-ts", truth_path,
            "-op", os.path.dirname(self.results_path) + "/"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Parse results
        results = []
        result_file = os.path.join(os.path.dirname(self.results_path), "Ball_Tree.out")
        
        # The format of the output file depends on the C++ code
        # This is a placeholder - you'll need to adapt based on actual output format
        with open(result_file, 'r') as f:
            # Skip header lines
            for _ in range(5):  # Adjust based on actual output
                next(f)
            
            for i in range(qn):
                line = next(f).strip()
                # Parse the line to extract k nearest neighbors
                # This is highly dependent on the output format
                neighbors = [int(x) for x in line.split(',')[:k]]
                results.append(neighbors)
        
        # Clean up temporary files
        for path in [self.data_path, self.query_path, truth_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return results
    
    def __str__(self) -> str:
        return f"BallTree(leaf_size={self.leaf_size})"


class BallTree(BaseANN):
    def __init__(self, metric: str, leaf_size: int = 1000):
        self._metric = metric
        self._leaf_size = leaf_size
        self._index = None
        
    def fit(self, X: np.ndarray) -> None:
        self._index = BallTreeWrapper(metric=self._metric, leaf_size=self._leaf_size)
        self._index.fit(X)
        
    def query(self, X: np.ndarray, k: int) -> List[List[int]]:
        return self._index.query(X, k)
        
    def __str__(self) -> str:
        return str(self._index)
