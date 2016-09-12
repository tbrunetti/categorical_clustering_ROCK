# categorical_clustering_ROCK

All the files and scripts in this directory are made to cluster categorical data using the unsupervised learning ROCK algorithm (RObust Clustering using linKs) from Guha, Rastogi, Shim, 1999 (http://dl.acm.org/citation.cfm?id=847264).

#### Dependencies
------------------
In order to run the entire suite of modules the following dependencies are required:
* numpy (http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
* scikit-learn (http://scikit-learn.org/stable/)
* pandas (http://pandas.pydata.org/pandas-docs/stable/install.html)

#### Installation
------------------
In order to run cat_cluster_ROCK.py, clone this repository into desired directory.  No further installation is required, excluding installation of dependencies above.

#### Running cat_cluster_ROCK.py
--------------------------------
To run the algorithm with the most basic parameters use the following:
```
cd categorical_clustering_ROCK
python cat_cluster_ROCK.py -input /path/to/input/matrix
```
The file format of the input should be a tab-delimited file, where each row represents an individual/point and each column represents an attribute/feature.  The number of desired clusters will default to 2, with a default similarity threshold of 0.5.  These default parameters can be overridden if -kclusters and -threshold flags are used in the python call followed by an integer and float, respectively.  For more information, use the -h/--help flags in the python call.

