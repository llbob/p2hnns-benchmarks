import os
import random
import tarfile
from urllib.request import build_opener, install_opener, urlopen, urlretrieve
import traceback
import time, gdown
import h5py
import numpy
from typing import Any, Callable, Dict, Tuple

# Needed for Cloudflare's firewall
opener = build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
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
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["points"][0])
    return hdf5_file, dimension


def construct_p2h_dataset(
    X: numpy.ndarray, test_size: int = 10000, dimension: int = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
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


def write_output(
    points: numpy.ndarray,
    hyperplanes: Tuple[numpy.ndarray, numpy.ndarray],
    fn: str,
    distance: str,
    point_type: str = "float",
    count: int = 100,
) -> None:
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


def glove(out_fn: str, d: int, distance: str) -> None:
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
        points, hyperplanes = construct_p2h_dataset(numpy.array(X))
        write_output(points, hyperplanes, out_fn, distance)


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
        points = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        # test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        hyperplanes = create_hyperplanes(points)
        write_output(points, hyperplanes, out_fn, "euclidean")


def cifar10(out_fn: str, distance: str) -> None:
    import tarfile
    import pickle
    import numpy as np

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    fn = os.path.join("data", "cifar-10-python.tar.gz")
    download(url, fn)

    # proces cifar10 data
    print("preparing %s" % out_fn)
    with tarfile.open(fn, "r:gz") as t:
        X = []
        # cifar10 consistof 5 batches of data
        for i in range(1, 6):
            batch_file = t.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            batch_data = pickle.load(batch_file, encoding="bytes")
            X.append(batch_data[b"data"])

        # concat the batches
        X = numpy.vstack(X).astype(numpy.float32)
        
        # NOTE: cifar10 is in rgb data range [0, 255] so can simply be normalized to [0, 1] by dividing by 255..

        points, hyperplanes = construct_p2h_dataset(X)
        write_output(points, hyperplanes, out_fn, distance)

def deep_download(src, dst=None, max_size=None): # credit to big-ann benchmarks for this function allowing us to download just a part of the file easily
    """download an URL, possibly cropped"""
    if os.path.exists(dst):
        print("Already exists")
        return
    print("downloading %s -> %s..." % (src, dst))
    if max_size is not None:
        print("   stopping at %d bytes" % max_size)
    t0 = time.time()
    outf = open(dst, "wb")
    inf = urlopen(src)
    info = dict(inf.info())
    content_size = int(info["Content-Length"])
    bs = 1 << 20
    totsz = 0
    while True:
        block = inf.read(bs)
        elapsed = time.time() - t0
        print(
            "  [%.2f s] downloaded %.2f MiB / %.2f MiB at %.2f MiB/s   "
            % (elapsed, totsz / 2**20, content_size / 2**20, totsz / 2**20 / elapsed),
            flush=True,
            end="\r",
        )
        if not block:
            break
        if max_size is not None and totsz + len(block) >= max_size:
            block = block[: max_size - totsz]
            outf.write(block)
            totsz += len(block)
            break
        outf.write(block)
        totsz += len(block)
    print()
    print(
        "download finished in %.2f s, total size %d bytes" % (time.time() - t0, totsz)
    )


def deep1m(out_fn: str, distance: str) -> None:
    import struct
    import numpy as np
    
    # the url for the full base deep1b ds
    url = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"
    fn = os.path.join("data", "deep1m.fbin")
    
    # calculate the maximum size to download: header (8 bytes) + 1m vectors of 96 dimensions (float32)
    max_size = 8 + (96 * 1_000_000 * 4)
    
    # use the download function from big-ann benchmarks to download just the first 1m vectors in this case
    deep_download(url, fn, max_size)
    
    print("preparing %s" % out_fn)
    
    # read the first 1m vectors from bin file
    with open(fn, "rb") as f:
        # read header
        nvecs, dim = struct.unpack('<ii', f.read(8))
        print(f"Original dataset has {nvecs} vectors of dimension {dim}")

        # again, to be sure, set the number of vectors to read
        count = 1_000_000
        
        # read into a numpy array
        vectors = np.fromfile(f, dtype=np.float32, count=count*dim)
        vectors = vectors.reshape(count, dim)
        
        print(f"Loaded {count} vectors of dimension {dim}")
    
    points, hyperplanes = construct_p2h_dataset(vectors)
    write_output(points, hyperplanes, out_fn, distance)


def music100(out_fn: str, distance: str) -> None:
    """
    Download and process the Music100 dataset from Google Drive.

    Args:
        out_fn (str): The output file name
        distance (str): The distance metric to use
    """
    import gdown
    import numpy as np

    # google drive shared link from folder p2hnns-benchmarks (https://drive.google.com/drive/folders/1LjyGXs881JhKIsJc4jou5-qo_WhQhaLQ?usp=sharing) last checked 20/03/2025
    url = "https://drive.google.com/file/d/1n_uwPyWw8JeODAV8-GEr_Ib6eTSW_4AP/view?usp=drive_link"

    # regex to extract file id from url
    file_id = url.split("/d/")[1].split("/view")[0]
    direct_url = f"https://drive.google.com/uc?id={file_id}"

    bin_fn = os.path.join("data", "database_music100.bin")

    # dl the file if it doesn't exist
    if not os.path.exists(bin_fn):
        print(f"downloading from google drive -> {bin_fn}...")
        try:
            # Use gdown for downloading from Google Drive
            gdown.download(direct_url, bin_fn, quiet=False)
        except:
            # If gdown is not installed, provide instructions
            print("Error: Could not download the file. Consider a manual download or make sure gdown is correctly installed.")
            return

    print(f"preparing {out_fn}")

    # read the bin file according to the authors of the datasets specifications
    databaseSize = 10**6
    dimension = 100
    try:
        data = np.fromfile(bin_fn, dtype=np.float32).reshape(databaseSize, dimension)

        points, hyperplanes = construct_p2h_dataset(data)
        write_output(points, hyperplanes, out_fn, distance)

    except Exception as e:
        print(f"Error processing the binary file: {e}")

def gist(out_fn: str, distance: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        points = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        # test = _get_irisa_matrix(t, "gist/gist_query.fvecs")
        hyperplanes = create_hyperplanes(points)
        write_output(points, hyperplanes, out_fn, distance)

def trevi(out_fn: str, distance: str) -> None:
    import zipfile
    from PIL import Image
    import numpy as np
    
    # url for the trevi dataset
    url = "https://phototour.cs.washington.edu/patches/trevi.zip"
    
    zip_fn = os.path.join("data", "trevi.zip")
    
    # dl file if it doesn't exist
    download(url, zip_fn)
    
    # create a temp directory to use for extracting fles
    extract_dir = os.path.join("data", "trevi_temp")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    print(f"Extracting {zip_fn} to {extract_dir}...")
    with zipfile.ZipFile(zip_fn, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # find all BMP image files which are represented as patches in the dataset
    bmp_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.bmp'):
                bmp_files.append(os.path.join(root, file))
    
    print(f"Found {len(bmp_files)} BMP patch files")
    
    # load the info.txt file which contains metadata about the patches
    info_path = os.path.join(extract_dir, "info.txt")
    
    # init list to store all patch vectors in
    patch_vectors = []
    
    # process each BMP file.. open, convert to numpy array, extract patches and flatten.NOTE: again it's rgb, so normalizing to [0, 1] by dividing by 255.
    for bmp_file in bmp_files:
        img = Image.open(bmp_file)
        img_array = np.array(img)
        # process all patches in the img from left to right and 
        # each patch is 64x64 pixels
        patch_size = 64
        height, width = img_array.shape
        
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                if y + patch_size <= height and x + patch_size <= width:
                    patch = img_array[y:y+patch_size, x:x+patch_size]
                    vector = patch.flatten().astype(np.float32)
                    patch_vectors.append(vector)
    
    # make ds by convert to numpy array
    points = np.array(patch_vectors)
    
    # create hyperplanes
    print(f"Creating hyperplanes from {len(points)} points...")
    hyperplanes = create_hyperplanes(points)
    
    # write the processed data to output file
    print(f"Writing output to {out_fn}")
    write_output(points, hyperplanes, out_fn, distance)
    
    # clean up temp directory
    import shutil
    shutil.rmtree(extract_dir)
    
    print(f"Trevi dataset processing complete. Output saved to {out_fn}")


DATASETS: Dict[str, Callable[[str], None]] = {
    "glove-25-angular": lambda out_fn: glove(out_fn, 25, "angular"),
    "glove-25-euclidean": lambda out_fn: glove(out_fn, 25, "euclidean"),
    # "glove-100-angular": lambda out_fn: glove(out_fn, 100, "angular"),
    "glove-100-euclidean": lambda out_fn: glove(out_fn, 100, "euclidean"),
    "cifar-10-euclidean": lambda out_fn: cifar10(out_fn, "euclidean"),
    # "cifar-10-angular": lambda out_fn: cifar10(out_fn, "angular"),
    "sift-128-euclidean": sift,
    "deep1m-euclidean": lambda out_fn: deep1m(out_fn, "euclidean"),
    "music100-euclidean": lambda out_fn: music100(out_fn, "euclidean"),
    # "music100-angular": lambda out_fn: music100(out_fn, "angular"),
    "trevi-euclidean": lambda out_fn: trevi(out_fn, "euclidean"),
    "gist-euclidean": lambda out_fn: gist(out_fn, "euclidean"),
}
