# The VampPrior Mixture Model

This code generates all results in our submitted manuscript.

## Package Requirements

Our ``Dockerfile`` creates the computational environment from TensorFlow's official image for version 2.12.0 with GPU support and uses ``requirements.txt`` to install additionally required packages.

## Reproducing Experiments

Run ``elbo_surgery_run.sh`` to produce Table 1.

Run ``clustering_experiments_run.sh`` to produce Table 2 and Figures 1, 2, and 4.

Run ``single_cell_experiments_run.sh`` to produce Tables 3 and 4 and Figures 3, 5-8.

## Repository Overview

``priors.py`` implements the DP GMM and VMM.

``models.py`` has the VAE models.

``single_cell_models.py`` has our implementation of scVI (Lopez et al., 2018).

``clustering_*.py`` are used by ``clustering_experiments_run.sh``.

``elbo_surgery_*.py`` are used by ``elbo_surgery_run.sh``.

``single_cell_*.py`` are used by ``single_cell_experiments_run.sh``.

