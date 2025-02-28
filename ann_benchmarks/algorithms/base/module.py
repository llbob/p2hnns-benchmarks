from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import psutil

import numpy

class BaseANN(object):
    """Base class/interface for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking."""

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        pass

    def get_memory_usage(self) -> Optional[float]:
        """Returns the current memory usage of this ANN algorithm instance in kilobytes.

        Returns:
            float: The current memory usage in kilobytes (for backwards compatibility), or None if
                this information is not available.
        """

        return psutil.Process().memory_info().rss / 1024

    def index(self, X: numpy.array) -> None:
        """Fits the ANN algorithm to the provided data. 

        Note: This is a placeholder method to be implemented by subclasses.

        Args:
            X (numpy.array): The data to index the algorithm to.
        """
        pass

    def query(self, q: numpy.array, b: float, n: int) -> numpy.array:
        """Performs a hyperplane query on the algorithm to find the nearest neighbors.

        Args:
            q (numpy.array): The normal vector of the hyperplane.
            b (float): The bias of the hyperplane.
            n (int): The number of nearest neighbors to return.

        Returns:
            numpy.array: An array of indices representing the nearest neighbors.
        """
        return []  # array of candidate indices

    def batch_query(self, normals: numpy.array, biases: numpy.array, n: int) -> None:
        """Performs multiple hyperplane queries at once.

        Args:
            normals (numpy.array): Array of normal vectors.
            biases (numpy.array): Array of biases.
            n (int): The number of nearest neighbors to return for each query.
        """
        pool = ThreadPool()
        self.res = pool.map(lambda args: self.query(args[0], args[1], n), 
                            zip(normals, biases))

    def get_batch_results(self) -> numpy.array:
        """Retrieves the results of a batch query (from .batch_query()).

        Returns:
            numpy.array: An array of nearest neighbor results for each query in the batch.
        """
        return self.res

    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}

    def __str__(self) -> str:
        return self.name