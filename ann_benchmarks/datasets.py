import os
import random
import tarfile
from urllib.request import build_opener, install_opener, urlopen, urlretrieve
import traceback

import h5py
import numpy
from typing import Any, Callable, Dict, Tuple

# Needed for Cloudflare's firewall
opener = build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
install_opener(opener)


def download(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.
    
    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(destination_path):
        print(f"downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)


def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str) -> Tuple[h5py.File, int]:
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading, 
    and the file handle and the dimension of the dataset are returned.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name)
    try:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
        download(dataset_url, hdf5_filename)
    except:
        traceback.print_exc()
        print(f"Cannot download {dataset_url}")
        if dataset_name in DATASETS:
            print("Creating dataset locally")
            DATASETS[dataset_name](hdf5_filename)

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension


def construct_p2h_dataset(X: numpy.ndarray, test_size: int = 10000, dimension: int = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Construct a dataset of points and hyperplanes.
    
    Args:
        X (numpy.ndarray): Input data array
        test_size (int, optional): The number of samples to include in the test set. 
            Defaults to 10000.
        dimension (int, optional): The dimensionality of the data. If not provided, 
            it will be inferred from the second dimension of X. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: A tuple containing:
            - points: The input data points
            - hyperplanes: A tuple of (normals, biases) defining the hyperplanes
    """
    points = X
    hyperplanes = create_hyperplanes(points)
    return points, hyperplanes

def create_hyperplanes(X: numpy.ndarray, n_hyperplanes: int = 10000) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Create random hyperplanes and biases using numpy. 
    
    Args:
        X (numpy.ndarray): Input data array
        n_hyperplanes (int, optional): The number of hyperplanes to create. Defaults to 10000.
    
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Hyperplane normals and their biases
    """
    # Generate all random indices at once
    idx = numpy.random.randint(X.shape[0], size=(n_hyperplanes, 3))
    
    # Get all random points at once using fancy indexing
    rand_points = X[idx]  # Shape: (n_hyperplanes, 3, dimension)
    
    # Calculate hyperplane normals (rand_1 - rand_2)
    normalvectors = rand_points[:, 0] - rand_points[:, 1]  # Shape: (n_hyperplanes, dimension)
    
    # Calculate biases using dot product
    biases = numpy.sum(normalvectors * rand_points[:, 2], axis=1)  # Shape: (n_hyperplanes,)
    
    return normalvectors, biases
    

def write_output(points: numpy.ndarray, hyperplanes: Tuple[numpy.ndarray, numpy.ndarray], fn: str, distance: str, point_type: str = "float", count: int = 100) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes 
    and stores the nearest neighbors and their distances for the test set using a 
    brute-force approach.
    
    Args:
        points (numpy.ndarray): The points.
        hyperplanes (Tuple[numpy.ndarray, numpy.ndarray]): A tuple of (hyperplane normals, hyperplane biases) defining the hyperplanes.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for 
            each point in the test set. Defaults to 100.
    """
    from ann_benchmarks.algorithms.bruteforce.module import BruteForceBLAS

    normals, biases = hyperplanes
    
    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(points[0])
        f.attrs["point_type"] = point_type
        print(f"points size: {points.shape[0]} * {points.shape[1]}")
        print(f"hyperplane normals size:  {normals.shape[0]} * {normals.shape[1]}")
        f.create_dataset("points", data=points)
        f.create_dataset("normals", data=normals)
        f.create_dataset("biases", data=biases)

        # Create datasets for neighbors and distances
        neighbors_ds = f.create_dataset("neighbors", (len(normals), count), dtype=int)
        distances_ds = f.create_dataset("distances", (len(normals), count), dtype=float)

        # Fit the brute-force k-NN model
        bf = BruteForceBLAS(distance, precision=points.dtype)
        bf.index(points)

        for i, (normal, bias) in enumerate(zip(normals, biases)):
            if i % 1000 == 0:
                print(f"{i}/{len(normals)}...")

            # Query the model with both normal and bias
            res = list(bf.query_with_distances(normal, bias, count))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors_ds[i] = [idx for idx, _ in res]
            distances_ds[i] = [dist for _, dist in res]


"""
param: train and test are arrays of arrays of indices.
"""




def train_test_split(X: numpy.ndarray, test_size: int = 10000, dimension: int = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Splits the provided dataset into a training set and a testing set.
    
    Args:
        X (numpy.ndarray): The dataset to split.
        test_size (int, optional): The number of samples to include in the test set. 
            Defaults to 10000.
        dimension (int, optional): The dimensionality of the data. If not provided, 
            it will be inferred from the second dimension of X. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the training set and the testing set.
    """
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    dimension = dimension if not None else X.shape[1]
    print(f"Splitting {X.shape[0]}*{dimension} into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)


def glove(out_fn: str, d: int) -> None:
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        # X_train, X_test = train_test_split(X)
        points, hyperplanes = construct_p2h_dataset(numpy.array(X))
        write_output(points, hyperplanes, out_fn, "angular")


def _load_texmex_vectors(f: Any, n: int, k: int) -> numpy.ndarray:
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str) -> numpy.ndarray:
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("data", "sift.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def gist(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        test = _get_irisa_matrix(t, "gist/gist_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def _load_mnist_vectors(fn: str) -> numpy.ndarray:
    import gzip
    import struct

    print("parsing vectors in %s..." % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d"),
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0] for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0] for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn: str) -> None:
    download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "mnist-train.gz")  # noqa
    download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "mnist-test.gz")  # noqa
    train = _load_mnist_vectors("mnist-train.gz")
    test = _load_mnist_vectors("mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


def fashion_mnist(out_fn: str) -> None:
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-train.gz",
    )
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",  # noqa
        "fashion-mnist-test.gz",
    )
    train = _load_mnist_vectors("fashion-mnist-train.gz")
    test = _load_mnist_vectors("fashion-mnist-test.gz")
    write_output(train, test, out_fn, "euclidean")


# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.
def deep_image(out_fn: str) -> None:
    yadisk_key = "https://yadi.sk/d/11eDCm7Dsn9GA"
    response = urlopen(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
        + yadisk_key
        + "&path=/deep10M.fvecs"
    )
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(",")[0][9:-1]
    filename = os.path.join("data", "deep-image.fvecs")
    download(dataset_url, filename)

    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = numpy.fromfile(filename, dtype=numpy.float32)
    dim = fv.view(numpy.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]

    X_train, X_test = train_test_split(fv)
    write_output(X_train, X_test, out_fn, "angular")


def transform_bag_of_words(filename: str, n_dimensions: int, out_fn: str) -> None:
    import gzip

    from scipy.sparse import lil_matrix
    from sklearn import random_projection
    from sklearn.feature_extraction.text import TfidfTransformer

    with gzip.open(filename, "rb") as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:]  # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(n_components=n_dimensions).fit_transform(B)
        X_train, X_test = train_test_split(C)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")


def nytimes(out_fn: str, n_dimensions: int) -> None:
    fn = "nytimes_%s.txt.gz" % n_dimensions
    download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz", fn
    )  # noqa
    transform_bag_of_words(fn, n_dimensions, out_fn)


def random_float(out_fn: str, n_dims: int, n_samples: int, centers: int, distance: str) -> None:
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1)
    X_train, X_test = train_test_split(X, test_size=0.1)
    write_output(X_train, X_test, out_fn, distance)


def dbpedia_entities_openai_1M(out_fn, n = None):
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset
    import numpy as np

    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")
    if n is not None and n >= 100_000:
        data = data.select(range(n))

    embeddings = data.to_pandas()['openai'].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))

    X_train, X_test = train_test_split(embeddings, test_size=10_000, random_state=42)

    write_output(X_train, X_test, out_fn, "angular")

def coco(out_fn: str, kind: str):
    assert kind in ('t2i', 'i2i')

    local_fn = "coco-clip-b16-512-features.hdf5"
    url = "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/%s" % local_fn
    download(url, local_fn)

    with h5py.File(local_fn, "r") as f:
        img_X = f['img_feats'][:]

        X_train, X_test = train_test_split(img_X, test_size=10_000)

        if kind == 't2i':
            # there are 5 captions per image, take the first one
            txt_X = f['txt_feats'][::5]
            _, X_test = train_test_split(txt_X, test_size=10_000)

    write_output(X_train, X_test, out_fn, "angular")


DATASETS: Dict[str, Callable[[str], None]] = {
    "deep-image-96-angular": deep_image,
    "fashion-mnist-784-euclidean": fashion_mnist,
    "gist-960-euclidean": gist,
    "glove-25-angular": lambda out_fn: glove(out_fn, 25),
    "glove-50-angular": lambda out_fn: glove(out_fn, 50),
    "glove-100-angular": lambda out_fn: glove(out_fn, 100),
    "glove-200-angular": lambda out_fn: glove(out_fn, 200),
    "mnist-784-euclidean": mnist,
    "random-xs-20-euclidean": lambda out_fn: random_float(out_fn, 20, 10000, 100, "euclidean"),
    "random-s-100-euclidean": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "euclidean"),
    "random-xs-20-angular": lambda out_fn: random_float(out_fn, 20, 10000, 100, "angular"),
    "random-s-100-angular": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "angular"),
    "sift-128-euclidean": sift,
    "nytimes-256-angular": lambda out_fn: nytimes(out_fn, 256),
    "nytimes-16-angular": lambda out_fn: nytimes(out_fn, 16),
    "coco-i2i-512-angular": lambda out_fn: coco(out_fn, "i2i"),
    "coco-t2i-512-angular": lambda out_fn: coco(out_fn, "t2i"),
}

DATASETS.update({
    f"dbpedia-openai-{n//1000}k-angular": lambda out_fn, i=n: dbpedia_entities_openai_1M(out_fn, i)
    for n in range(100_000, 1_100_000, 100_000)
})
