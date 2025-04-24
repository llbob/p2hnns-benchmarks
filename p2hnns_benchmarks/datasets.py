import os
import random
import struct
import tarfile
from urllib.request import build_opener, install_opener, urlopen, urlretrieve
import traceback
import time, gdown
import h5py
import numpy as np
import zipfile
from typing import Any, Callable, Dict, Tuple
from sklearn.decomposition import PCA
from p2hnns_benchmarks import generate_queries

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

def create_hyperplanes_rpsd(X: np.ndarray, n_hyperplanes: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    random point sample distance (rpsd) ?
    This method generates hyperplanes by randomly selecting five points from the dataset and using them to:

    create normal vectors by taking the difference between the mean of two 5 point samples of points. 
    calculate biases using randomly generated factors that are 

    Args:
        X (np.ndarray): Input data array
        n_hyperplanes (int, optional): The number of hyperplanes to create. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Hyperplane normals and their biases
    """
    # Generate all random indices at once
    idx = np.random.randint(X.shape[0], size=(n_hyperplanes, 10))

    # Get all random points at once using fancy indexing
    rand_points = X[idx]  # Shape: (n_hyperplanes, 10, dimension)

    # Calculate the mean of the first two samples
    first_sample_mean = np.mean(rand_points[:, :5], axis=1)  # Shape: (n_hyperplanes, dimension)
    second_sample_mean = np.mean(rand_points[:, 5:10], axis=1)  # Shape: (n_hyperplanes, dimension)

    # Calculate hyperplane normals (rand_1 - rand_2)
    normalvectors = first_sample_mean - second_sample_mean
    
    # Check for zero norm vectors to avoid division by zero in distance calculations later on
    norms = np.linalg.norm(normalvectors, axis=1)
    # `np.finfo(np.float32).eps` gives the machine epsilon for float32
    zero_indices = np.where(norms < np.finfo(np.float32).eps)[0]
    
    # Replace zero norm vectors if any
    if len(zero_indices) > 0:
        for i in zero_indices:
            # Get new random points until we get a non-zero normal vector
            while norms[i] < np.finfo(np.float32).eps:
                new_idx = np.random.randint(X.shape[0], size=2)
                normalvectors[i] = X[new_idx[0]] - X[new_idx[1]]
                norms[i] = np.linalg.norm(normalvectors[i])
                
    # Normalize vectors to unit length
    unit_normals = normalvectors / norms[:, np.newaxis]
    
    # generate random factors between -0.2 and 1.2
    # this determines where the hyperplane will be positioned relative to the two sample means
    factors = np.random.uniform(-0.2, 1.2, n_hyperplanes)
    
    # now calculate the points through which the hyperplanes will pass
    # when factor = 0.0: point is at second_sample_mean
    # factor = 1.0: point is at first_sample_mean
    # factor = 0.5 point is halfway between
    points_between_samples = second_sample_mean + factors[:, np.newaxis] * normalvectors
    
    # For a point p to lie on the hyperplane, we need: unit_normal · p + bias = 0.
    # and solving for bias: bias = -(unit_normal · p)
    # What we effectively get is that the hyperplanes passes through these points between the samples..
    biases = -np.sum(unit_normals * points_between_samples, axis=1)
    
    return unit_normals, biases

def create_hyperplanes_bctree(X: np.ndarray, n_hyperplanes: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    (bctree) method to create hyperplanes and biases.
    
    The implementation follows a similar approach to Qiang et als implementation in https://github.com/HuangQiang/BC-Tree
    
    Args:
        X (np.ndarray): Input data array
        n_hyperplanes (int, optional): The number of hyperplanes to create. Defaults to 1000.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Hyperplane normals and their biases
    """
    from p2hnns_benchmarks.generate_queries.module import generate_hyperplanes
    
    # generate the hyperplanes using the C++ implementation
    normals, biases = generate_hyperplanes(X, n_hyperplanes)
    
    return normals, biases

def write_output(
    points: np.ndarray,
    hyperplanes: Tuple[np.ndarray, np.ndarray],
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
        points (np.ndarray): The points.
        hyperplanes (Tuple[np.ndarray, np.ndarray]): A tuple of (hyperplane normals, hyperplane biases) defining the hyperplanes.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for
            each point in the test set. Defaults to 100.
    """
    from p2hnns_benchmarks.algorithms.bruteforce.module import BruteForceBLAS

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


def glove(out_fn: str, d: int, distance: str, size: int = None, hyperplane_method:str = "rpsd") -> None:
    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(np.array(v))
        X = X[:size] if size is not None else X
        points = np.array(X)
        if hyperplane_method == "rpsd":
            hyperplanes = create_hyperplanes_rpsd(points)
        elif hyperplane_method == "bctree":
            hyperplanes = create_hyperplanes_bctree(points)
        else:
            raise ValueError(f"unknown hyperplane method: {hyperplane_method}")
        write_output(points, hyperplanes, out_fn, distance)


def _load_texmex_vectors(f: Any, n: int, k: int) -> np.ndarray:
    v = np.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str) -> np.ndarray:
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
        hyperplanes = create_hyperplanes_rpsd(points)
        write_output(points, hyperplanes, out_fn, "euclidean")


def cifar10(out_fn: str, distance: str, size: int = None, hyperplane_method: str = "rpsd") -> None:
    import tarfile
    import pickle
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    fn = os.path.join("data", "cifar-10-python.tar.gz")
    download(url, fn)

    # proces cifar10 data
    print("preparing %s" % out_fn)
    with tarfile.open(fn, "r:gz") as t:
        X = []

        # extract 
        # cifar10 consistof 5 batches of data
        for i in range(1, 6):
            batch_file = t.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            batch_data = pickle.load(batch_file, encoding="bytes")
            # get the labels
            # get the labels
            # labels.extend(batch_data[b"labels"])  # Added to collect labels
            X.append(batch_data[b"data"])

        # concat the batches
        X = np.vstack(X).astype(np.float32)

        # labels = np.array(labels, dtype=np.int32)  # convert labels to numpy array

        X = X[:size] if size is not None else X

        # apply PCA to reduce the dimensionality to 512 from the original 3072(32x32x3), this seems to be a standard for the benchmarks we're comparing to here in the field of p2hnns
        pca = PCA(n_components=512, random_state=42)
        X = pca.fit_transform(X)

        points = np.array(X)
        # labels = np.array(labels)
        
        if hyperplane_method == "rpsd":
            hyperplanes = create_hyperplanes_rpsd(points)
        elif hyperplane_method == "bctree":
            hyperplanes = create_hyperplanes_bctree(points)
        # elif hyperplane_method == "svm-basic":
        #     hyperplanes = create_hyperplanes_svm(points, labels)
        # elif hyperplane_method == "svm-advanced":
        #     hyperplanes = create_hyperplanes_svm(points, labels, advanced=True)
        else:
            raise ValueError(f"unknown hyperplane method: {hyperplane_method}")

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


def deepm(out_fn: str, distance: str, count:int) -> None: 
    if count == 1_000_000:
        name = "deep1m"
    elif count == 10_000_000:
        name = "deep10m"
    elif count == 100_000_000:
        name = "deep100m"
    else:
        raise ValueError("Unsupported count value. Only 1m, 10m and 100m supported.")
    
    # the url for the full base deep1b ds
    url = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"
    fn = os.path.join("data", f"{name}.fbin")
    
    # calculate the maximum size to download: header (8 bytes) + 1m vectors of 96 dimensions (float32)
    max_size = 8 + (96 * count * 4)
    
    # use the download function from big-ann benchmarks to download just the first 1m vectors in this case
    deep_download(url, fn, max_size)
    
    print("preparing %s" % out_fn)
    
    # read the first 1m vectors from bin file
    with open(fn, "rb") as f:
        # read header
        nvecs, dim = struct.unpack('<ii', f.read(8))
        print(f"Original dataset has {nvecs} vectors of dimension {dim}")

        # again, to be sure, set the number of vectors to read
        count = count
        
        # read into a numpy array
        X = np.fromfile(f, dtype=np.float32, count=count*dim)
        X = points.reshape(count, dim)
        
        print(f"Loaded {count} vectors of dimension {dim}")
    
    points = np.array(X)
    hyperplanes = create_hyperplanes_rpsd(points)
    write_output(points, hyperplanes, out_fn, distance)


def music100(out_fn: str, distance: str) -> None:
    """
    Download and process the Music100 dataset from Google Drive.

    Args:
        out_fn (str): The output file name
        distance (str): The distance metric to use
    """
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
        X = np.fromfile(bin_fn, dtype=np.float32).reshape(databaseSize, dimension)

        points = np.array(X)
        hyperplanes = create_hyperplanes_rpsd(points)
        write_output(points, hyperplanes, out_fn, distance)

    except Exception as e:
        print(f"Error processing the binary file: {e}")

def gist(out_fn: str, distance: str, size: int = None) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        X = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        X = X[:size] if size is not None else X
        points = np.array(X)
        hyperplanes = create_hyperplanes_rpsd(points)
        write_output(points, hyperplanes, out_fn, distance)

def trevi(out_fn: str, distance: str, size: int = None) -> None:
    
    from PIL import Image
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
    
    # process each BMP file.. open, convert to numpy array, extract patches and flatten. 
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
    patch_vectors = patch_vectors[:size] if size is not None else patch_vectors
    points = np.array(patch_vectors)
    hyperplanes = create_hyperplanes_rpsd(points)
    write_output(points, hyperplanes, out_fn, distance)
    
    # clean up temp directory
    import shutil
    shutil.rmtree(extract_dir)

DATASETS: Dict[str, Callable[[str], None]] = {
    # ========================================================================
    # Here are datasets that are used as testers - all 'med' covers datasets of 20k points only
    "glove-25-euclidean-20k": lambda out_fn: glove(out_fn, 25, "euclidean", 20000),
    "glove-25-angular-20k": lambda out_fn: glove(out_fn, 25, "angular", 20000),
    "glove-100-euclidean-20k": lambda out_fn: glove(out_fn, 100, "euclidean", 20000),
    "glove-100-angular-20k": lambda out_fn: glove(out_fn, 100, "angular", 20000),
    "cifar10-512-euclidean-20k": lambda out_fn: cifar10(out_fn, "euclidean", 20000),
    "gist-960-euclidean-20k": lambda out_fn: gist(out_fn, "euclidean", 20000),
    "trevi-4096-euclidean-20k": lambda out_fn: trevi(out_fn, "euclidean", 20000),

    # ========================================================================
    # Here are the datasets that are used to demonstrate the hyperplane methods
    # 25 dims - 20k points
    "glove-25-euclidean-20k-bctree": lambda out_fn: glove(out_fn, 25, "euclidean", 20000, hyperplane_method="bctree"),
    "glove-25-euclidean-20k-rpsd": lambda out_fn: glove(out_fn, 25, "euclidean", 20000, hyperplane_method="rpsd"),
    # 100 dims - 20k points
    "glove-100-euclidean-20k-bctree": lambda out_fn: glove(out_fn, 100, "euclidean", 20000, hyperplane_method="bctree"),
    "glove-100-euclidean-20k-rpsd": lambda out_fn: glove(out_fn, 100, "euclidean", 20000, hyperplane_method="rpsd"),

    # ========================================================================
    # Here are the datasets that are not currently used
    "glove-25-euclidean": lambda out_fn: glove(out_fn, 25, "euclidean"),
    "glove-25-angular": lambda out_fn: glove(out_fn, 25, "angular"),
    "deep10m-96-euclidean": lambda out_fn: deepm(out_fn, "euclidean", 10_000_000),
    "glove-100-euclidean": lambda out_fn: glove(out_fn, 100, "euclidean"),
    "glove-100-angular": lambda out_fn: glove(out_fn, 100, "angular"),
    "music-100-euclidean": lambda out_fn: music100(out_fn, "euclidean"),
    "sift-128-euclidean": sift,
    "cifar10-512-euclidean": lambda out_fn: cifar10(out_fn, "euclidean"),
    "gist-960-euclidean": lambda out_fn: gist(out_fn, "euclidean"),
    "trevi-4096-euclidean": lambda out_fn: trevi(out_fn, "euclidean"),
}
