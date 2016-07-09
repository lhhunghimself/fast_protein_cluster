#!/bin/sh
../fast_protein_cluster -i list -o output --prune_rmsd --prune_until_size 300 --cluster_tmscore --nclusters 10 --nthreads 8 --kmeans --pvalue .01 --percentile .99 --gpu
