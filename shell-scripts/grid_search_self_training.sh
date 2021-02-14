#!/bin/bash

#sizes=(0.90 0.95 0.98 0.99) #0.98 0.99)
sizes=(50 100 200)
for k in "${sizes[@]}"
do
	sbatch -o logs/xlmr-large-sarcasm-normalized-large-st-$k shell-scripts/run_st_exp_large.sh $k
done
