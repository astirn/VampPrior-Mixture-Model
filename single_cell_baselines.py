import argparse
import gc
import numpy as np
import os
import pandas as pd
import scanpy.pp as pp
import scipy.sparse as sparse
import zlib

from anndata import AnnData
from datasets import load_sc_dataset
from scanpy.external.pp import harmony_integrate, scanorama_integrate
from scib_metrics.benchmark import Benchmarker
from single_cell_models import NoBatchCorrection


def additional_baselines(data: AnnData, seed: int = 0) -> AnnData:
    data.X = data.X.todense() if sparse.issparse(data.X) else data.X
    data.X = np.asarray(data.X)
    pp.normalize_total(data)
    pp.log1p(data)
    pp.scale(data)
    pp.pca(data, random_state=seed)

    data.obs['batch'] = data.obs['batch'].astype(str)
    sorted_indices = np.argsort(data.obs['batch'])
    data = data[sorted_indices].copy()

    harmony_integrate(data, adjusted_basis='Harmony_N/A_', key='batch', random_state=seed)
    scanorama_integrate(data, adjusted_basis='Scanorama_N/A_', key='batch', verbose=1)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pbmc')
    parser.add_argument('--seed', type=int, default=80085)
    parser.add_argument('--num_trials', type=int, default=10)
    args = parser.parse_args()

    results_dir = os.path.join('experiments', 'scRNA-seq', str(args.seed), args.dataset)
    integration_file = os.path.join(results_dir, '{:s}_integration_performance.pkl'.format('testing'))
    integration = pd.read_pickle(integration_file) if os.path.exists(integration_file) else pd.DataFrame()

    for trial in range(1, args.num_trials + 1):
        trial_seed = int(zlib.crc32(str(trial * (args.seed or 1)).encode())) % (2 ** 32 - 1)

        anndata = load_sc_dataset(args.dataset)[0]
        single_batch = anndata.obs['batch'].nunique() == 1

        anndata = additional_baselines(anndata, seed=trial_seed)

        bm = Benchmarker(
            anndata,
            batch_key='batch',
            label_key='cell_type',
            embedding_obsm_keys=['Harmony_N/A_', 'Scanorama_N/A_'],
            batch_correction_metrics=NoBatchCorrection() if single_batch else None,
            n_jobs=-1,
        )
        bm.benchmark()
        df = bm.get_results(min_max_scale=False).copy()

        df.columns = pd.MultiIndex.from_arrays([df.loc['Metric Type'], df.columns])
        df = df.loc[~df.index.isin(['Metric Type'])].astype(float)
        if single_batch:
            df = df[[col for col in df.columns if 'Bio conservation' in col]]
        index = [i.split('_', maxsplit=2) + [trial] for i in df.index.values]
        df.index = pd.MultiIndex.from_tuples(index, names=['Model', 'Prior', 'Config.', 'Trial'])
        integration = pd.concat([integration, df])

        del anndata, bm, df
        gc.collect()

    integration.to_pickle(integration_file.replace('.pkl', '_additional.pkl'))