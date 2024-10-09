# The VampPrior Mixture Model

This code generates all results in [our manuscript](https://arxiv.org/abs/2402.04412):
```
@misc{stirn2024vampprior,
      title={The VampPrior Mixture Model}, 
      author={Andrew Stirn and David A. Knowles},
      year={2024},
      eprint={2402.04412},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-vampprior-mixture-model/image-clustering-on-fashion-mnist)](https://paperswithcode.com/sota/image-clustering-on-fashion-mnist?p=the-vampprior-mixture-model)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-vampprior-mixture-model/unsupervised-image-classification-on-mnist)](https://paperswithcode.com/sota/unsupervised-image-classification-on-mnist?p=the-vampprior-mixture-model)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-vampprior-mixture-model/image-clustering-on-mnist-full)](https://paperswithcode.com/sota/image-clustering-on-mnist-full?p=the-vampprior-mixture-model)

## Package Requirements

Our ``Dockerfile`` creates the computational environment from TensorFlow's official image for version 2.15.0 with GPU support and uses ``requirements.txt`` to install additionally required packages.

## Reproducing Experiments

Run ``clustering_experiments_run.sh`` to produce Table 1 and Figures 1 and 3.

Run ``elbo_surgery_run.sh`` to produce Table 2.

Run ``single_cell_experiments_run.sh`` to produce Tables 3 and 4 and Figures 2, 4-6.

## Repository Overview

``priors.py`` implements the our Bayesian GMM and VMM.

``models.py`` contains the VAE models.

``single_cell_models.py`` has our implementation of scVI (Lopez et al., 2018).

``clustering_*.py`` are used by ``clustering_experiments_run.sh``.

``elbo_surgery_*.py`` are used by ``elbo_surgery_run.sh``.

``single_cell_*.py`` are used by ``single_cell_experiments_run.sh``.

