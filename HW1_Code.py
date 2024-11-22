import pandas as pd


## Define hyper-parameter ranges for D-tree
min_impurity_thr = [0.001,0.0001]
min_samples_split_thr = [5,10]
max_depth_thr = [3,5]
min_samples_leaf_thr = [3,5]
ccp_thr = [0.001,0.0001]