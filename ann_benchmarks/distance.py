from typing import Callable, List, NamedTuple, Tuple, Union

import h5py
import numpy as np

def norm(a):
    return np.sum(a**2) ** 0.5


def euclidean(a, b):
    return norm(a - b)

class Metric(NamedTuple):
    distance: Callable[[np.ndarray, np.ndarray], float]
    distance_valid: Callable[[float], bool]

metrics = {
    "euclidean": Metric(
        distance=lambda a, b: euclidean(a, b),
        distance_valid=lambda a: True
    ),
    "angular": Metric(
        distance=lambda a, b: 1 - np.dot(a, b) / (norm(a) * norm(b)),
        distance_valid=lambda a: True
    ),
}

def compute_distance(metric: str, a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the distance between two points according to a specified metric.

    Args:
        metric (str): The name of the metric to use. Must be a key in the 'metrics' dictionary.
        a (np.ndarray): The first point.
        b (np.ndarray): The second point.

    Returns:
        float: The computed distance.

    Raises:
        KeyError: If the specified metric is not found in the 'metrics' dictionary.
    """
    if metric not in metrics:
        raise KeyError(f"Unknown metric '{metric}'. Known metrics are {list(metrics.keys())}")

    return metrics[metric].distance(a, b)


def is_distance_valid(metric: str, distance: float) -> bool:
    """
    Check if a computed distance is valid according to a specified metric.

    Args:
        metric (str): The name of the metric to use. Must be a key in the 'metrics' dictionary.
        distance (float): The computed distance to check.

    Returns:
        bool: True if the distance is valid, False otherwise.

    Raises:
        KeyError: If the specified metric is not found in the 'metrics' dictionary.
    """
    if metric not in metrics:
        raise KeyError(f"Unknown metric '{metric}'. Known metrics are {list(metrics.keys())}")

    return metrics[metric].distance_valid(distance)





def dataset_transform(dataset: h5py.Dataset) -> Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    """
    Transforms the dataset from the HDF5 format to conventional numpy format.

    A dense dataset is returned as a numpy array.

    Args:
        dataset (h5py.Dataset): The input dataset in HDF5 format.

    Returns:
        Tuple: (points, hyperplanes).
    """
    
    points = np.array(dataset["points"])
    normals = np.array(dataset["normals"])
    biases = np.array(dataset["biases"])
    return points, (normals, biases)
    

