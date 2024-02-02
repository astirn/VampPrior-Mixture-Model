#!/bin/bash

SEED=42823
python3 clustering_experiments.py --seed $SEED --dataset "mnist" --latent_dim 10 --mode "tuning"
python3 clustering_experiments.py --seed $SEED --dataset "mnist" --latent_dim 10 --mode "testing"
python3 clustering_experiments.py --seed $SEED --dataset "fashion_mnist" --latent_dim 30 --mode "tuning"
python3 clustering_experiments.py --seed $SEED --dataset "fashion_mnist" --latent_dim 30 --mode "testing"
python3 clustering_analysis.py --seed $SEED