import argparse
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple, List, Union

import colors
import docker
import numpy
import psutil

from p2hnns_benchmarks.algorithms.base.module import BaseANN

from .definitions import Definition, instantiate_algorithm
from .datasets import DATASETS, get_dataset
from .distance import dataset_transform, metrics
from .results import store_results


def run_individual_query(algo: BaseANN, X_points: numpy.ndarray, 
                         hyperplanes: Tuple[numpy.ndarray, numpy.ndarray], 
                         distance: str, count: int, run_count: int, batch: bool) -> Tuple[dict, list]:
    """Run a search query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated P2HNNS algorithm.
        X_points (numpy.array): The points data.
        hyperplanes (Tuple[numpy.ndarray, numpy.ndarray]): Tuple of (normals, biases) for hyperplane queries.
        distance (str): The type of distance metric to use.
        count (int): The number of nearest neighbors to return.
        run_count (int): The number of times to run the query.
        batch (bool): Flag to indicate whether to run in batch mode or not.

    Returns:
        tuple: A tuple with the attributes of the algorithm run and the results.
    """
    normals, biases = hyperplanes
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
        (not batch) and hasattr(algo, "prepare_query")
    )

    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(normal: numpy.array, bias: float) -> Tuple[float, List[Tuple[int, float]]]:
            """Executes a single query on an instantiated, P2HNNS algorithm.

            Args:
                normal (numpy.array): Normal vector of the hyperplane.
                bias (float): Bias of the hyperplane.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: Tuple containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            if prepared_queries:
                algo.prepare_query(normal, bias, count)
                start = time.time()
                algo.run_prepared_query()
                total = time.time() - start
                candidates = algo.get_prepared_query_results()
            else:
                start = time.time()
                candidates = algo.query(normal, bias, count)
                total = time.time() - start

            # make sure all returned indices are unique
            assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = [
                (int(idx), float(metrics[distance].distance(X_points[idx], (normal, bias))))
                for idx in candidates
            ]
            
            n_items_processed[0] += 1
            if n_items_processed[0] % 250 == 0:
                print("Processed %d/%d queries..." % (n_items_processed[0], len(normals)))
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count"
                    " is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)

        def batch_query(normals_batch: numpy.array, biases_batch: numpy.array) -> List[Tuple[float, List[Tuple[int, float]]]]:
            """Executes a batch of hyperplane queries on an instantiated P2HNNS algorithm.

            Args:
                normals_batch (numpy.array): Array of normal vectors.
                biases_batch (numpy.array): Array of biases.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: List of tuples, each containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to hyperplane)
            """
            if prepared_queries:
                algo.prepare_batch_query(normals_batch, biases_batch, count)
                start = time.time()
                algo.run_batch_query()
                total = time.time() - start
            else:
                start = time.time()
                algo.batch_query(normals_batch, biases_batch, count)
                total = time.time() - start
                
            results = algo.get_batch_results()
            if hasattr(algo, "get_batch_latencies"):
                batch_latencies = algo.get_batch_latencies()
            else:
                batch_latencies = [total / float(len(normals_batch))] * len(normals_batch)

            # make sure all returned indices are unique
            for res in results:
                assert len(res) == len(set(res)), "Implementation returned duplicated candidates"

            candidates = [
                [(int(idx), float(metrics[distance].distance(X_points[idx], (normal, biases_batch[i]))))
                 for idx in single_results]  # noqa
                for normal, single_results in zip(normals_batch, results)
            ]
            return [(latency, v) for latency, v in zip(batch_latencies, candidates)]

        if batch:
            results = batch_query(normals, biases)
        else:
            results = [single_query(normal, bias) for normal, bias in zip(normals, biases)]

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(normals)
        avg_candidates = total_candidates / len(normals)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def load_and_transform_dataset(dataset_name: str) -> Tuple[
        numpy.ndarray,
        Tuple[numpy.ndarray, numpy.ndarray],
        str]:
    """Loads and transforms the dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple: Transformed datasets.
        points: The data points to index
        hyperplanes: A tuple of (normals, biases) for querying
        distance: The distance metric to use
    """
    D, dimension = get_dataset(dataset_name)

    points = numpy.array(D["points"])
    normals = numpy.array(D["normals"])
    biases = numpy.array(D["biases"])
    hyperplanes = (normals, biases)
    
    print(f"Got a points set of size ({points.shape[0]} * {dimension})")
    print(f"Got {normals.shape[0]} hyperplane queries")
    
    distance = D.attrs["distance"]
    return points, hyperplanes, distance



def build_index(algo: BaseANN, X_points: numpy.ndarray) -> Tuple:
    """Builds the ANN index for a given ANN algorithm on the training data.

    Args:
        algo (Any): The algorithm instance.
        X_train (Any): The training data.

    Returns:
        Tuple: The build time and index size.
    """
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    algo.index(X_points)
    build_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_usage_before

    print("Built index in", build_time)
    print("Index size: ", index_size)

    return build_time, index_size


def run(definition: Definition, dataset_name: str, count: int, run_count: int, batch: bool) -> None:
    """Run the algorithm benchmarking for point-to-hyperplane queries.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        count (int): The number of results to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
    """
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"""\
error: query argument groups have been specified for {definition.module}.{definition.constructor}({definition.arguments}), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function"""

    X_points, hyperplanes, distance = load_and_transform_dataset(dataset_name)

    try:
        if hasattr(algo, "supports_prepared_queries"):
            algo.supports_prepared_queries()

        build_time, index_size = build_index(algo, X_points)

        query_argument_groups = definition.query_argument_groups or [[]]  # Ensure at least one iteration

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            
            descriptor, results = run_individual_query(algo, X_points, hyperplanes, distance, count, run_count, batch)

            descriptor.update({
                "build_time": build_time,
                "index_size": index_size,
                "algo": definition.algorithm,
                "dataset": dataset_name
            })

            store_results(dataset_name, count, definition, query_arguments, descriptor, results, batch)
    finally:
        algo.done()

def run_from_cmdline():
    """Calls the function `run` using arguments from the command line. See `ArgumentParser` for 
    arguments, all run it with `--help`.
    """
    parser = argparse.ArgumentParser(
        """

            NOTICE: You probably want to run.py rather than this script.

"""
    )
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to benchmark on.", required=True)
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument(
        "--module", help='Python module containing algorithm. E.g. "p2hnns_benchmarks.algorithms.annoy"', required=True
    )
    parser.add_argument("--constructor", help='Constructer to load from modulel. E.g. "Annoy"', required=True)
    parser.add_argument(
        "--count", help="K: Number of nearest neighbours for the algorithm to return.", required=True, type=int
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If flag included, algorithms will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument("build", help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries. E.g. [100]", nargs="*", default=[])
    args = parser.parse_args()

    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
    )
    run(definition, args.dataset, args.count, args.runs, args.batch)


def run_docker(
    definition: Definition,
    dataset: str,
    count: int,
    runs: int,
    timeout: int,
    batch: bool,
    cpu_limit: str,
    mem_limit: Optional[int] = None
) -> None:
    """Runs `run_from_cmdline` within a Docker container with specified parameters and logs the output.

    See `run_from_cmdline` for details on the args.
    """
    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--count",
        str(count),
    ]
    if batch:
        cmd += ["--batch"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available

    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath("/var/run/docker.sock"): {"bind": "/var/run/docker.sock", "mode": "rw"},
            os.path.abspath("p2hnns_benchmarks"): {"bind": "/home/app/p2hnns_benchmarks", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
        },
        network_mode="host",
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        privileged=True,
        detach=True,
    )
    logger = logging.getLogger(f"annb.{container.short_id}")

    logger.info(
        "Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s"
        % (container.short_id, cpu_limit, mem_limit, timeout, cmd)
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except Exception as e:
        logger.error("Container.wait for container %s failed with exception", container.short_id)
        logger.error(str(e))
    finally:
        logger.info("Removing container")
        container.remove(force=True)


def _handle_container_return_value(
    return_value: Union[Dict[str, Union[int, str]], int],
    container: docker.models.containers.Container,
    logger: logging.Logger
) -> None:
    """Handles the return value of a Docker container and outputs error and stdout messages (with colour).

    Args:
        return_value (Union[Dict[str, Union[int, str]], int]): The return value of the container.
        container (docker.models.containers.Container): The Docker container.
        logger (logging.Logger): The logger instance.
    """

    base_msg = f"Child process for container {container.short_id} "
    msg = base_msg + "returned exit code {}"

    if isinstance(return_value, dict):  # The return value from container.wait changes from int to dict in docker 3.0.0
        error_msg = return_value.get("Error", "")
        exit_code = return_value["StatusCode"]
        msg = msg.format(f"{exit_code} with message {error_msg}")
    else:
        exit_code = return_value
        msg = msg.format(exit_code)

    if exit_code not in [0, None]:
        for line in container.logs(stream=True):
            logger.error(colors.color(line.decode(), fg="red"))
        logger.error(msg)
    else:
        logger.info(msg)
