# LBLD-algorithm

Local Balanced Label Diffusion Algorithm for Community Detection in Social Networks
This code implements LBLD algorithm, a fast and non-overlapping community detection algorithm in Python 3.7.

We have used neighboring list structure as the input file for algorithm. The execution of the LBLD is so simple.it is only needed to write the name of the dataset to run the algorithm.

Everything is built to run the algorithm (such as pathes to folders and files) and it is only needed to put the extracted folders of: "datasets","groundtruth" in one folder with the source code. in order to save the labels of nodes to a file, it is needed to create "results" folder in the same folder of source code.

To execute algorithm jupyter notebook or any other platform  which runs python can be used. The main configurations of LBLD are as follows:

```
# ---------------------- Configurations -------------------
dataset_name = "karate"     # name of dataset
path = "./datasets/" + dataset_name + ".txt"    # path to dataset
iteration = 1            # number of iterations for label selection step (mostly is set to 1 or 2)
merge_flag = 1           # merge_flag=0 -> do not merge //  merge_flag=1 -> do merge
write_flag = 0           # 1 means write nodes labels to file. 0 means do not write
modularity_flag = 1      # 1 means calculate modularity. 0 means do not calculate modularity
NMI_flag = 1             # 1 means calculate NMI. 0 means do not calculate NMI
# ---------------------------------------------------------
```

Names of datasets are as follows and are available in "datasets" folder.

# Datasets:

karate

Dolphins

Polbooks

Football

Netscience

power

ca_grqc

collaboration

ca_hepth

PGP

condmat_2003

condmat_2005

brightkite

DBLP

Amazon

Youtube
