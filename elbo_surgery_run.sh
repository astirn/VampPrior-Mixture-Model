#!/bin/bash

SEED=42823
python3 elbo_surgery.py --seed $SEED --dataset "fashion_mnist" --latent_dim 30