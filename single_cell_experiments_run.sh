#!/bin/bash

# random number seed
SEED=42823

# loop over datasets
DATASETS=("cortex" "pbmc" "lung atlas" "split-seq")
for dataset in "${DATASETS[@]}"; do
  for i in $(seq 1 5);
  do
    python3 single_cell_experiments.py --seed $SEED --dataset "$dataset" --mode "tuning" --trial $i
  done
  for i in $(seq 1 10);
  do
    python3 single_cell_experiments.py --seed $SEED --dataset "$dataset" --mode "testing" --trial $i
  done
done

# analyze results
python3 single_cell_analysis.py --seed $SEED
