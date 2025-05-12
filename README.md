Benchmarking P2HNNS Algorithms
==============================
P2HNNS-benchmarks is a benchmarking environment for approximate nearest neighbor search algorithms. This website contains the current benchmarking results. Please visit P2HNNS-benchmarks to get an overview of the evaluated datasets and algorithms. Make a pull request on GitHub to add your own code or improvements to the benchmarking system. We acknowledge and give full credit to the original ANN-BENCHMARKS repository, developed by Martin Aumueller, Erik Bernhardsson, and Alec Faitfull, from which this project is forked.

This is currently a work in progress. We are still working on the documentation, the code and the results made available on https://p2hnns-benchmarks.com/.

... W.I.P. ...

Have python 3.10 installed.

pip install -r requirements.txt

python install.py

python create_dataset.py --dataset <dataset_name> # dataset_name as found in datasets.py

python run.py # runs just glove-100-euclidean on all datasets

python create_website.py # creates the website, can be used with --outputdir <outputdir> to specify the output directory, and --latex to create latex plots, --timeout <timeout> to specify the timeout for the algorithms, --groupplots to generate latex groupplots in the outputdir folder picker and much more..

___

Further this repo has been expanded with scripts for calculating expansion, local relative contrast (RC) and local intrinsic dimensionality (LID), the inspiration as well as the scripts(where only RC have been adapted to work for P2HNNS) come from Martin Aumüller and Matteo Ceccarello from the paper "The Role of Local Dimensionality Measures in Benchmarking Nearest Neighbor Search" and their github repo (role-of-dimensionality)[https://github.com/Cecca/role-of-dimensionality].

To make use of them:

RC:
python compute-rc.py data/<dataset_name>.hdf5 > <dataset_name>-rc.txt

LID:
python compute-lid.py data/<dataset_name>.hdf5 > <dataset_name>-lid.txt

Expansion:
python compute-expansion.py data/<dataset_name>.hdf5 > <dataset_name>-expansion.txt

Or tather use our adaptation of them generating plots for predefined datasets through: 
python compute-rc-lid-expansion.py