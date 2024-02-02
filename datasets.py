import anndata
import os
import scvi

import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_datasets as tfds


def load_sc_dataset(dataset: str):
    if dataset == 'brain large':
        data = scvi.data.brainlarge_dataset()
        data.obs.rename(columns={'labels': 'cell_type'}, inplace=True)
    elif dataset == 'cortex':
        data = scvi.data.cortex()
        sc.pp.highly_variable_genes(data, n_top_genes=558, flavor='seurat_v3', subset=True)
        data.obs['batch'] = 0
    elif dataset == 'pbmc':
        data = scvi.data.pbmc_dataset()
        data.obs.rename(columns={'str_labels': 'cell_type'}, inplace=True)
    elif dataset == 'lung atlas':
        data = sc.read(
            os.path.join('data', 'lung_atlas.h5ad'),
            backup_url="https://figshare.com/ndownloader/files/24539942",
        )
        sc.pp.highly_variable_genes(data, n_top_genes=2000, flavor='seurat_v3', subset=True, batch_key='batch')
        _, data.obs['batch'] = np.unique(data.obs['batch'], return_inverse=True)
    elif dataset == 'retina':
        data = scvi.data.retina()
        data.obs.rename(columns={'labels': 'cell_type'}, inplace=True)
    elif dataset == 'split-seq':
        meta = pd.read_csv('data/split-seq/cells_meta.tsv.gz', sep='\t')
        meta.rename(columns={'polydT': 'batch', 'cellchat_clusters': 'cell_type'}, inplace=True)
        data = anndata.AnnData(
            X=sparse.load_npz('data/split-seq/counts_small.npz'),
            obs=meta[['batch', 'cell_type']]
        )
    else:
        raise NotImplementedError
    sc.pp.filter_genes(data, min_counts=3)
    sc.pp.filter_cells(data, min_counts=3)
    data.layers['counts'] = data.X.todense() if sparse.issparse(data.X) else data.X.copy()
    data.obs['batch'] = data.obs['batch'].astype(int)

    # load to GPU
    counts = tf.constant(data.layers['counts'])
    batch_id = tf.one_hot(data.obs['batch'], depth=data.obs['batch'].nunique())
    labels = tf.constant(np.unique(data.obs['cell_type'], return_inverse=True)[1])

    return data, counts, batch_id, labels


def greedy_maximum_coverage(batch_id, labels):
    num_batches = batch_id.shape[1]
    num_types = tf.shape(tf.unique(labels)[0])[0].numpy()

    # number of cell type (columns) counts per donor (rows)
    types_per_batch = [tf.boolean_mask(labels, batch_id[:, i]).numpy() for i in range(num_batches)]
    counts = np.array([[sum(types == i) for i in range(num_types)] for types in types_per_batch])
    counts = counts.astype(float)

    # greedily filter cell types
    included_types = set()
    done = False
    old_score = counts.sum()
    while not done:

        # count matrix with excluded batches removed
        excluded_batches = set(np.where(np.any(counts[:, list(included_types)] == 0, axis=1))[0].tolist())
        adjusted_counts = counts[list(set(range(num_batches)) - excluded_batches)]

        # current score
        old_score = adjusted_counts[:, list(included_types)].sum()

        # new score
        adjusted_counts[:, list(included_types)] = 0
        proposed_types = set(np.where((adjusted_counts > 0).all(axis=0))[0])
        if len(proposed_types) == 0:
            proposed_types = {np.argmax(adjusted_counts.sum(axis=0))}
            excluded_batches = excluded_batches.union(set(np.where(counts[:, list(proposed_types)] == 0)[0].tolist()))
        adjusted_counts = counts[list(set(range(num_batches)) - excluded_batches)]
        new_score = adjusted_counts[:, list(included_types.union(proposed_types))].sum()

        # stopping logic
        if new_score > old_score:
            included_types = included_types.union(proposed_types)
        else:
            done = True

    # indices of the subset
    included_batches = np.where(np.all(counts[:, list(included_types)] > 0, axis=1))[0]
    filtered_batch_indices = np.in1d(tf.argmax(batch_id, axis=1).numpy(), included_batches)
    filtered_type_indices = np.in1d(labels.numpy(), list(included_types))
    filtered_indices = np.where(np.logical_and(filtered_batch_indices, filtered_type_indices))[0]
    assert len(filtered_indices) == old_score

    return filtered_indices


def load_tensorflow_dataset(dataset):
    ds_train = tfds.load(name=dataset, split=tfds.Split.TRAIN, data_dir='data')
    ds_valid = tfds.load(name=dataset, split=tfds.Split.TEST, data_dir='data')
    x_train, labels_train = [tuple(ele.values()) for ele in ds_train.batch(len(ds_train))][0]
    x_valid, labels_valid = [tuple(ele.values()) for ele in ds_valid.batch(len(ds_valid))][0]
    x_train = tf.cast(x_train, tf.float32)
    x_valid = tf.cast(x_valid, tf.float32)
    max_val = tf.reduce_max(x_train)
    x_train = (x_train / max_val - 0.5) * 2
    x_valid = (x_valid / max_val - 0.5) * 2

    return x_train, labels_train, x_valid, labels_valid


if __name__ == '__main__':

    # download all the VAE data
    for vae_dataset in ['mnist', 'fashion_mnist', 'svhn_cropped']:
        load_tensorflow_dataset(vae_dataset)
