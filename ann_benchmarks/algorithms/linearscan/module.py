import os
import numpy
import subprocess
import re
import time
from typing import Any, Dict, Optional, List, Tuple
import psutil
from ann_benchmarks.algorithms.base import BaseANN

class LinearScan(BaseANN):
    """P2HNNS Linear Scan implementation for hyperplane queries."""
    
    def __init__(self, metric="angular", params=None):
        """Initialize the P2H Linear Scan algorithm."""
        self.name = "linearscan"
        self.metric = metric
        self.params = params or {}
        
        # Set paths
        self.p2h_path = "/home/app/p2h-ann/methods"
        self.data_dir = "/home/app/p2h-ann/data/bin"
        self.results_dir = "/home/app/p2h-ann/results"
        
        # Create temp dataset name
        self.dataset_name = "temp_dataset"
        self.data_path = os.path.join(self.data_dir, self.dataset_name)
        self.results_path = os.path.join(self.results_dir, self.dataset_name)
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        self.indexed = False
        self.res = []
        self.query_times = []
        self.index_time = 0.0
        
    def index(self, X: numpy.array) -> None:
        """Index the data for P2H Linear Scan."""
        start_time = time.time()
        self.n, self.d = X.shape
        print(f"Indexing {self.n} points of dimension {self.d}")
        
        # Save data to binary file
        X = X.astype(numpy.float32)
        with open(f"{self.data_path}.ds", "wb") as f:
            f.write(X.tobytes())
            
        self.indexed = True
        self.index_time = time.time() - start_time
        print(f"Indexed data saved to {self.data_path}.ds in {self.index_time:.2f} seconds")
        
    def query(self, q: numpy.array, b: float, k: int) -> numpy.array:
        """Query the P2H Linear Scan algorithm."""
        if not self.indexed:
            raise RuntimeError("Index the data first before querying")
        
        # For P2H, we need to create a query with the normal vector and bias
        # The hyperplane is defined as q·x + b = 0
        query = numpy.append(q, b).astype(numpy.float32).reshape(1, self.d)
        
        # Save query to binary file
        with open(f"{self.data_path}.q", "wb") as f:
            f.write(query.tobytes())
            
        # First, generate ground truth if it doesn't exist
        if not os.path.exists(f"{self.data_path}.gt"):
            gt_cmd = [
                os.path.join(self.p2h_path, "p2h"),
                "-alg", "0",  # Ground Truth
                "-n", str(self.n),
                "-qn", "1",
                "-d", str(self.d),
                "-dt", "float32",
                "-dn", self.dataset_name,
                "-ds", f"{self.data_path}.ds",
                "-qs", f"{self.data_path}.q",
                "-ts", f"{self.data_path}.gt",
                "-op", f"{self.results_path}/"
            ]
            print(f"Running ground truth command: {' '.join(gt_cmd)}")
            subprocess.run(gt_cmd, check=True, cwd=self.p2h_path)
        
        # Run P2H Linear Scan
        ls_cmd = [
            os.path.join(self.p2h_path, "p2h"),
            "-alg", "1",  # Linear Scan
            "-n", str(self.n),
            "-qn", "1",
            "-d", str(self.d),
            "-dt", "float32",
            "-dn", self.dataset_name,
            "-ds", f"{self.data_path}.ds",
            "-qs", f"{self.data_path}.q",
            "-ts", f"{self.data_path}.gt",
            "-op", f"{self.results_path}/"
        ]
        print(f"Running linear scan command: {' '.join(ls_cmd)}")
        subprocess.run(ls_cmd, check=True, cwd=self.p2h_path)
        
        # Parse results
        results_file = os.path.join(self.results_path, "p2h_linear_scan.out")
        indices, query_time = self._parse_results(results_file, k)
        self.query_times.append(query_time)
        
        return indices
        
    def _parse_results(self, results_file: str, k: int) -> Tuple[numpy.array, float]:
        """Parse results from the output file."""
        if not os.path.exists(results_file):
            print(f"Warning: Results file {results_file} not found")
            return numpy.array(list(range(min(k, self.n)))), 0.0
            
        try:
            # Read the results file
            with open(results_file, 'r') as f:
                content = f.read()
                
            # Extract query time
            time_match = re.search(r'Query Time: ([\d.]+) Seconds', content)
            query_time = float(time_match.group(1)) if time_match else 0.0
                
            # Extract results - this depends on the exact format of the output file
            # Assuming the format has lines with "Query i: id1,id2,id3,..."
            results_match = re.search(r'Query 1: ([\d,]+)', content)
            if results_match:
                indices_str = results_match.group(1).split(',')
                indices = [int(idx) for idx in indices_str[:k]]
                return numpy.array(indices), query_time
            else:
                # Alternative parsing: look for result lines
                result_lines = re.findall(r'(\d+)\s+(\d+)\s+([\d.]+)', content)
                if result_lines:
                    # Format might be: query_id result_id distance
                    indices = [int(line[1]) for line in result_lines[:k]]
                    return numpy.array(indices), query_time
                else:
                    print(f"Warning: Could not parse results from {results_file}")
                    return numpy.array(list(range(min(k, self.n)))), query_time
                
        except Exception as e:
            print(f"Error parsing results: {e}")
            return numpy.array(list(range(min(k, self.n)))), 0.0
        
    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result."""
        return {
            "query_times": self.query_times,
            "avg_query_time": numpy.mean(self.query_times) if self.query_times else 0.0,
            "index_time": self.index_time
        }        
    def done(self) -> None:
        """Clean up resources."""
        # Remove temporary files
        for ext in [".ds", ".q", ".gt"]:
            if os.path.exists(f"{self.data_path}{ext}"):
                try:
                    os.remove(f"{self.data_path}{ext}")
                except Exception as e:
                    print(f"Error removing {self.data_path}{ext}: {e}")
