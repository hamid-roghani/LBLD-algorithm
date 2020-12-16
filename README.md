# LBLD-algorithm
Local Balanced Label Diffusion Algorithm for Community Detection in Social Networks
this code implements LBLD algorithm in Python 3.7.
we have used neighboring list structure as the input file for algorithm.
the execution of the LBLD is so simple.it is only needed to write the name of the dataset to run the algorithm.
Datasets names can be selected from the "datasets" folder.
the main configurations to start execution of LBLD is as follows:
# --------------------- Configurations ---------------------
dataset_name = "karate" # name of dataset
path = "./datasets/" + dataset_name + ".txt" # path to dataset
iteration = 1           # number of iterations for label selection step (mostly is set to 1 or 2)
merge_flag = 1         # merge_flag=0 -> do not merge //  merge_flag=1 -> do merge
write_flag = 0        # 1 means write nodes labels to file. 0 means do not write
modularity_flag = 1  # 1 means calculate modularity. 0 means do not calculate modularity
NMI_flag = 1        # 1 means calculate NMI. 0 means do not calculate NMI
#----------------------------------------------------
Everything is built to run the algorithm (pathes to folders and files) and it is only needed to execute algorithm using jupyter note book.
