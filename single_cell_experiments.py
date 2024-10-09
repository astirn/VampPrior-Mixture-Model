import argparse
import ast
import os
import torch  # must import before any other package imports tensorflow or bash will have segmentation fault
import pickle
import priors
import pymde
import scvi
import zlib

import numpy as np
import pandas as pd
import single_cell_models as sc
import tensorflow as tf

from callbacks import PerformanceMonitor
from datasets import load_sc_dataset
from scib_metrics.benchmark import Benchmarker
from utils import clustering_performance

# configure GPUs for pyTorch
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    use_gpu = int(np.argmax([torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]))
else:
    use_gpu = False

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pbmc')
parser.add_argument('--max_clusters', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=10000)
parser.add_argument('--mode', type=str)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--replace', action='store_true', default=False, help='force saved model replacement')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--train_ratio', type=float, default=0.9)
parser.add_argument('--trial', type=int, default=1)
args = parser.parse_args()

# check arguments
assert isinstance(args.trial, int) and args.trial > 0
assert args.mode in {'tuning', 'testing'}

# make results directory
results_dir = os.path.join('experiments', 'scRNA-seq', str(args.seed), args.dataset)
os.makedirs(results_dir, exist_ok=True)

# load data and get model configuration
data, counts, batch_id, labels = load_sc_dataset(args.dataset)
model_config = sc.dataset_config(args.dataset)

# log library size mean and variance conditioned on batch
log_counts_batch = np.ma.log(tf.einsum('ij,ik->ik', tf.cast(counts, tf.float32), batch_id))
library_log_mean = np.mean(log_counts_batch, axis=0)
library_log_var = np.var(log_counts_batch, axis=0)

# tuning mode test cases
if args.mode == 'tuning':
    batch_sizes = [128, 256, 512]
    prior_learning_ratios = [1.0, 2.5, 5.0, 10.0]
    if model_config['learning_rate'] >= 1e-3:
        prior_learning_ratios = [0.1, 0.25, 0.5] + prior_learning_ratios
    test_cases = [
        dict(model='scVI',
             prior='VampPriorMixture',
             prior_kwargs=dict(inference='MAP-DP', prior_learning_ratio=plr, use_labels=False))
        for plr in prior_learning_ratios]

# testing mode test cases
elif args.mode == 'testing':
    try:
        # identify best tuning parameters
        df_int = pd.read_pickle(os.path.join(results_dir, 'tuning_integration_performance.pkl'))
        df_int.reset_index('Trial', drop=True, inplace=True)
        df_int = df_int.groupby(level=df_int.index.names).mean().loc['scVI']
        df_clust = pd.read_pickle(os.path.join(results_dir, 'tuning_clustering_performance.pkl'))
        df_clust.reset_index('Trial', drop=True, inplace=True)
        df_clust = df_clust.groupby(level=df_clust.index.names).mean()
        df_clust = df_clust[df_clust['Utilized clusters'] >= tf.unique(labels)[0].shape[0]]
        df_int = df_int.loc[df_int.index.isin(df_clust.index)]
        if ('Aggregate score', 'Total') in df_int.columns:
            best_config = df_int[('Aggregate score', 'Total')].idxmax()
        else:
            best_config = df_int[('Aggregate score', 'Bio conservation')].idxmax()
        best_config = ast.literal_eval(best_config[1])
        batch_sizes = [best_config['batch_size']]
        plr = best_config['prior_learning_ratio']
    except FileNotFoundError:
        raise FileNotFoundError('please run script in tuning mode first')

    # assemble test cases
    test_cases = [
        dict(model='scVI', prior='StandardNormal', prior_kwargs=dict()),
        dict(model='scVI', prior='VampPrior', prior_kwargs=dict()),
        dict(model='scVI', prior='GaussianMixture', prior_kwargs=dict(inference='MAP-DP', prior_learning_ratio=plr)),
        dict(model='scVI', prior='VampPriorMixture',
             prior_kwargs=dict(inference='MAP-DP', prior_learning_ratio=plr, use_labels=False)),
    ]

else:
    raise NotImplementedError

# load/initialize results
clustering_file = os.path.join(results_dir, '{:s}_clustering_performance.pkl'.format(args.mode))
clustering = pd.read_pickle(clustering_file) if os.path.exists(clustering_file) else pd.DataFrame()
integration_file = os.path.join(results_dir, '{:s}_integration_performance.pkl'.format(args.mode))
integration = pd.read_pickle(integration_file) if os.path.exists(integration_file) else pd.DataFrame()

# loop over the batch sizes
for i, batch_size in enumerate(batch_sizes):
    print('*** {:s} | Trial {:d} | BS {:d}/{:d} ***'.format(args.dataset, args.trial, i + 1, len(batch_sizes)))

    # trial directory
    trial_dir = os.path.join(results_dir, 'trial_{:d}'.format(args.trial))

    # a deterministic but seemingly random transformation of the provided seed into a trial seed
    trial_seed = int(zlib.crc32(str(args.trial * (args.seed or 1)).encode())) % (2 ** 32 - 1)

    # official scVI model
    model_name, prior, config = 'scVI-official', 'StandardNormal', dict(batch_size=batch_size)
    print('--- {:s} ---'.format(model_name))
    save_path = os.path.join(trial_dir, model_name, str(config))
    os.makedirs(save_path, exist_ok=True)
    try:
        # load results
        i_train = np.load(os.path.join(save_path, 'train_indices.npy'))
        i_valid = np.load(os.path.join(save_path, 'valid_indices.npy'))
        embeddings = np.load(os.path.join(save_path, 'embeddings.npy'))
        mde = np.load(os.path.join(save_path, 'mde.npy'))

    except FileNotFoundError:
        # configure and train
        if trial_seed is not None:
            torch.manual_seed(trial_seed)
        scvi.model.SCVI.setup_anndata(data, layer='counts', batch_key='batch')
        model = scvi.model.SCVI(
            adata=data,
            n_hidden=model_config['hidden_dim'],
            n_latent=model_config['latent_dim'],
            n_layers=model_config['n_layers'],
            gene_likelihood=model_config['likelihood'],
            deeply_inject_covariates=False,
            log_variational=True)
        model.train(
            max_epochs=args.max_epochs,
            train_size=args.train_ratio,
            batch_size=config['batch_size'],
            check_val_every_n_epoch=1,
            early_stopping=True,
            early_stopping_patience=args.patience,
            plan_kwargs=dict(
                lr=model_config['learning_rate'],
                weight_decay=0.0,
                eps=1e-7,
                lr_patience=args.max_epochs,
                n_steps_kl_warmup=0,
                n_epochs_kl_warmup=0
            )
        )

        # save results
        i_train = model.train_indices.copy()
        np.save(os.path.join(save_path, 'train_indices.npy'), i_train)
        i_valid = model.validation_indices.copy()
        np.save(os.path.join(save_path, 'valid_indices.npy'), i_valid)
        pymde.seed(trial_seed)
        embeddings = model.get_latent_representation()
        np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
        mde = scvi.model.utils.mde(embeddings)
        np.save(os.path.join(save_path, 'mde.npy'), mde)

        # clear model from memory
        torch.cuda.empty_cache()

    # store latent representation and plot its MDE
    data.obsm['_'.join([model_name, prior, str(config)])] = embeddings
    sc.plot_mde_of_embeddings(mde, data.obs['batch'], data.obs['cell_type'],
                              fig_path=os.path.join(save_path, 'mde.pdf'))

    # take the same train/validation splits from scVI official implementation
    train_data = dict(x=tf.gather(counts, i_train), s=tf.gather(batch_id, i_train))
    valid_data = dict(x=tf.gather(counts, i_valid), s=tf.gather(batch_id, i_valid))

    # loop over our priors
    for test_case in test_cases:
        config = {**dict(batch_size=batch_size), **test_case['prior_kwargs']}
        print('--- {:s} | {:s} : {:s} ---'.format(test_case['model'], test_case['prior'], str(config)))
        save_path = os.path.join(trial_dir, test_case['model'], test_case['prior'], str(config))

        # set the seed
        tf.keras.utils.set_random_seed(trial_seed)
        tf.config.experimental.enable_op_determinism()

        # select prior
        u = sc.vamp_prior_pseudo_inputs(
            count_matrix=counts,
            one_hot_batch_id=batch_id,
            num_clusters=args.max_clusters,
            cell_labels=labels if config.get('use_labels') else None)
        latent_prior = priors.select_prior(test_case['prior'], **config, **dict(
            latent_dim=model_config['latent_dim'],
            num_clusters=args.max_clusters,
            u=u,
            learning_rate=model_config['learning_rate'] * (config.get('prior_learning_ratio') or 0),
        ))

        # construct and compile model
        model = sc.scVI(
            n_genes=counts.shape[1],
            n_batches=batch_id.shape[1],
            prior=latent_prior,
            use_observed_library_size=True,
            library_log_loc=library_log_mean,
            library_log_scale=library_log_var ** 0.5,
            **model_config
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate']))

        # if we are set to resume and the model directory already contains a saved model, load it
        if not bool(args.replace) and os.path.exists(os.path.join(save_path, 'checkpoint')):
            print('Loading existing model.')
            checkpoint = tf.train.Checkpoint(model)
            checkpoint.restore(os.path.join(save_path, 'best_checkpoint')).expect_partial()

        # otherwise, fit and save the model
        else:
            hist = model.fit(
                x=train_data,
                validation_data=valid_data,
                batch_size=config['batch_size'],
                epochs=args.max_epochs,
                verbose=False,
                callbacks=[PerformanceMonitor(patience=args.patience)]
            )
            model.save_weights(os.path.join(save_path, 'best_checkpoint'))
            with open(os.path.join(save_path, 'history.pkl'), 'wb') as f:
                pickle.dump(hist.history, f)

        # get predictions (reset seed in case models were loaded)
        tf.keras.utils.set_random_seed(trial_seed)
        embeddings = model.predict(dict(x=counts, s=batch_id), batch_size=config['batch_size'])
        np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
        pymde.seed(trial_seed)
        mde = scvi.model.utils.mde(embeddings)
        cluster_probs = model.cluster_probabilities(embeddings)
        if cluster_probs.shape[1] > 1:
            np.save(os.path.join(save_path, 'cluster_probs.npy'), cluster_probs.numpy())

        # store latent representation and plot its MDE
        data.obsm['_'.join([test_case['model'], test_case['prior'], str(config)])] = embeddings
        sc.plot_mde_of_embeddings(mde, data.obs['batch'], data.obs['cell_type'],
                                  cluster=np.argmax(cluster_probs.numpy(), axis=1),
                                  fig_path=os.path.join(save_path, 'mde.pdf'))

        # save clustering performance
        index = pd.MultiIndex.from_arrays(
            arrays=[[test_case['prior']], [str(config)], [args.trial]],
            names=['Prior', 'Config.', 'Trial'])
        clustering = clustering.loc[~clustering.index.isin(index)]
        clustering = pd.concat([clustering, clustering_performance(cluster_probs, data.obs['cell_type'], index)])
        clustering.to_pickle(clustering_file)

        # clear model from memory
        tf.keras.backend.clear_session()

        # if tuning mode and number of utilized clusters less than true number continue
        if args.mode == 'tuning' and clustering.iloc[-1]['Utilized clusters'] < tf.unique(labels)[0].shape[0]:
            print('Cluster threshold reached!')
            break

# single batch only
single_batch = data.obs['batch'].nunique() == 1

# benchmark performance
bm = Benchmarker(
    data,
    batch_key='batch',
    label_key='cell_type',
    embedding_obsm_keys=[k for k in data.obsm.keys() if 'scVI' in k],
    batch_correction_metrics=sc.NoBatchCorrection() if single_batch else None,
    n_jobs=-1,
)
bm.benchmark()

# aggregate results
df = bm.get_results(min_max_scale=False).copy()
df.columns = pd.MultiIndex.from_arrays([df.loc['Metric Type'], df.columns])
df = df.loc[~df.index.isin(['Metric Type'])].astype(float)
if single_batch:
    df = df[[col for col in df.columns if 'Bio conservation' in col]]
index = [i.split('_', maxsplit=2) + [args.trial] for i in df.index.values]
df.index = pd.MultiIndex.from_tuples(index, names=['Model', 'Prior', 'Config.', 'Trial'])
integration = integration.loc[~integration.index.isin(index)]
integration = pd.concat([integration, df])
integration.to_pickle(integration_file)
