# Benchmarking P2HNNS Algorithms
==============================
P2HNNS-benchmarks is a benchmarking environment for point-to-hyperplane approximate nearest neighbor search algorithms, it's a fork of ANN-BENCHMARKS developed by Martin Aumueller, Erik Bernhardsson, and Alec Faitfull.

The results have been made available on https://p2hnns-benchmarks.com/.


We recommend using conda or pyenv to create a virtual environment running python 3.10 which is currently what we've been testing with.

Installation of the benchmarking setup
==============================

Have python 3.10 installed and cd into the repo, then run the following commands:

```
pip install -r requirements.txt
````

```
python install.py
``` 

Download and create the datasets:
```
python create_dataset.py --dataset <dataset_name>
```

Currently the supported datasets are:
- glove-25-euclidean
- deep10m-96-euclidean
- glove-100-euclidean 
- glove-200-euclidean 
- music-100-euclidean 
- sift-128-euclidean 
- cifar10-512-euclidean 
- fashion-mnist-784-euclidean
- gist-960-euclidean 
- trevi-4096-euclidean

Note in regards to hyperplane generation:
As a standard the 'point sample mean' method for generating hyperplanes will be used by default on these datasets. If you want to try out the 'gaussian random normal' check out the branch 'feature/queries-generation-wrapper'. In here you can can use the datasets 'glove-100-euclidean-psm' or 'glove-100-euclidean-grn' to test the two different methods of generating hyperplanes. The 'gaussian random normal' method generates hyperplanes using a wrapper of Huang Qiangs method for hyperplane generation in the file 'generate.cc' in https://github.com/HuangQiang/BC-Tree.

Running the benchmarks
==============================
To run the benchmarks, you can use the following command:

```
python run.py --dataset <dataset_name> --algorithm <algorithm_name>
````

Then to generate plots of the results, for latex too, and the website to view them on:
```
python create_website.py --latex --outputdir benchmark_results
```
You can also specify the output directory with `--outputdir <directory_name>`.

In some cases it may be necessary to expand the timeout for the benchmarks, you can do this with the `--timeout` flag eg.
```
python3 run.py --timeout 20000 --dataset <dataset_name> --algorithm <algorithm_name>
```


Additional analysis tools:
==============================
Further this repo has been expanded with scripts for calculating expansion, local relative contrast (RC) and local intrinsic dimensionality (LID), the inspiration as well as the scripts(where only RC have been adapted to work for P2HNNS) come from Martin Aumüller and Matteo Ceccarello from the paper "The Role of Local Dimensionality Measures in Benchmarking Nearest Neighbor Search" and their github repo (role-of-dimensionality)[https://github.com/Cecca/role-of-dimensionality].

To make use of them, update the script `compute-rc-lid-expansion.py` with the datasets you want to compute the metrics for, and run it with the command:
```
python compute-rc-lid-expansion.py
```