import argparse
import os
import scvi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from matplotlib import rcParams
from datasets import load_sc_dataset
from priors import clean_prior_name
from utils import process_results

# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


# table order
dataset_order = ['cortex', 'pbmc', 'split-seq', 'lung atlas']


def simplify_configurations(df):
    configs = df.index.get_level_values('Config.').unique()
    configs = configs[~configs.str.contains("'use_labels': True")]
    return df.loc[df.index.get_level_values('Config.').isin(configs)]


def clean_model_name(name: str) -> str:
    if name == 'scVI':
        return 'scVI (our code)'
    elif name == 'scVI-official':
        return '\\href{https://docs.scvi-tools.org/en/stable/user_guide/models/scvi.html}{scVI tools}'
    else:
        return name


def clean_table(df):
    if 'Model' in df.index.names:
        df.reset_index(['Model'], inplace=True)
        df['Model'] = df['Model'].apply(clean_model_name)
        df.set_index('Model', append=True, inplace=True)
    df.reset_index(['Prior'], inplace=True)
    df['Prior'] = df['Prior'].apply(clean_prior_name)
    df.set_index('Prior', append=True, inplace=True)
    if 'Config.' in df.index.names:
        df.reset_index('Config.', drop=True, inplace=True)
    df.fillna('--', inplace=True)
    return df


def plot_tuning(exp_path):
    # loop over datasets
    for dataset in os.listdir(exp_path):
        try:
            clustering = pd.read_pickle(os.path.join(exp_path, dataset, 'tuning_clustering_performance.pkl'))
            integration = pd.read_pickle(os.path.join(exp_path, dataset, 'tuning_integration_performance.pkl'))
        except FileNotFoundError:
            continue
        if len(clustering) == 0 or len(integration) == 0:
            continue

        # load dataset
        data, _, _, _ = load_sc_dataset(dataset)

        # only keep VMM
        clustering = clustering.loc[clustering.index.get_level_values('Prior') == 'VampPriorMixture']
        integration = integration.loc[integration.index.get_level_values('Prior') == 'VampPriorMixture']

        # extract batch size and prior learning rates
        for df in [clustering, integration]:
            df.reset_index(['Trial', 'Config.'], drop=False, inplace=True)
            df['Config.'] = df['Config.'].apply(lambda s: s.split(','))
            df['Batch size'] = df['Config.'].apply(lambda s: int(s[0].split(' ')[1]))
            df['Prior learning ratio'] = df['Config.'].apply(lambda s: float(s[2].split(' ')[2]))
            df.set_index(['Batch size', 'Prior learning ratio'], inplace=True)

        # find the best config and rename score
        if ('Aggregate score', 'Total') in integration.columns:
            integration['Score'] = integration[('Aggregate score', 'Total')]
        else:
            integration['Score'] = integration[('Aggregate score', 'Bio conservation')]

        # find the best score subject to utilized clusters > number of annotated cell types
        mean_clust = clustering['Utilized clusters'].groupby(level=clustering.index.names).mean()
        mean_clust = mean_clust.loc[mean_clust >= data.obs['cell_type'].nunique()]
        mean_score = integration['Score'].groupby(level=integration.index.names).mean()
        mean_score = mean_score.loc[mean_score.index.isin(mean_clust.index)]
        best_batch, best_plr = mean_score.idxmax()
        best_score = mean_score.max()

        # reset indices
        clustering.reset_index(inplace=True)
        integration.reset_index(inplace=True)

        # plot results
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        hue = 'Prior learning ratio'
        sns.lineplot(clustering, x='Batch size', y='Utilized clusters', hue=hue, marker='o', ax=ax[0])
        sns.move_legend(ax[0], loc='upper left', bbox_to_anchor=(0.0, 1.0))
        ax[0].set_xticks(clustering['Batch size'].unique())
        ax[0].set_yticks([y for y in np.arange(0, clustering['Utilized clusters'].max() + 10, 10)])
        ax[0].axhline(y=data.obs['cell_type'].nunique(), color='tab:red', linestyle=':')
        sns.lineplot(integration, x='Batch size', y='Score', hue=hue, legend=False, marker='o', ax=ax[1])
        ax[1].set_xticks(integration['Batch size'].unique())
        offset = 0.0175 * (max(ax[1].get_yticks()) - min(ax[1].get_yticks()))
        ax[1].plot(
            [best_batch - 8] + 2 * [best_batch + 8] + 2 * [best_batch - 8],
            2 * [best_score - offset] + 2 * [best_score + offset] + [best_score - offset],
            color='tab:green',
            linestyle=':')
        plt.tight_layout()
        fig.savefig(os.path.join('results', 'scRNA-tuning-{:s}.pdf'.format(dataset.replace(' ', '-'))))


def print_integration_table(exp_path, threshold):
    # loop over datasets
    integration_table = pd.DataFrame()
    for dataset in dataset_order:
        for filtered in [False]:
            try:
                if filtered:
                    dataset_name = dataset + ' (filtered)'
                    file_name = 'testing_integration_performance_filtered.pkl'
                else:
                    dataset_name = dataset
                    file_name = 'testing_integration_performance.pkl'
                integration = pd.read_pickle(os.path.join(exp_path, dataset, file_name))
            except FileNotFoundError:
                continue
            if len(integration) == 0:
                continue

            # integration table processing
            integration = simplify_configurations(integration)
            integration.reset_index('Trial', drop=True, inplace=True)
            integration = process_results(integration, threshold, pm_std=False)

            # update integration table
            integration['Dataset'] = dataset_name
            integration.set_index('Dataset', append=True, inplace=True)
            integration = integration.reorder_levels(['Dataset', 'Model', 'Prior', 'Config.'])
            integration_table = pd.concat([integration_table, integration])

    # print table
    integration_table.sort_index(axis=1, level=0, ascending=True, inplace=True)
    s = clean_table(integration_table).style.format(precision=2)
    col_fmt = 'l' * integration_table.index.nlevels
    metric_type = None
    for col in integration_table.columns:
        if col[0] != metric_type:
            col_fmt += '|'
            metric_type = col[0]
        col_fmt += 'c'
    s.to_latex(os.path.join('results', 'scRNA-integration.tex'),
               column_format=col_fmt,
               multicol_align='c|',
               hrules=True,
               clines="skip-last;data",
               multirow_align='t')


def print_clustering_table(exp_path, threshold):
    # loop over datasets
    clustering_table = pd.DataFrame()
    for dataset in dataset_order:
        try:
            clustering = pd.read_pickle(os.path.join(exp_path, dataset, 'testing_clustering_performance.pkl'))
        except FileNotFoundError:
            continue
        if len(clustering) == 0:
            continue

        # load dataset
        data, _, _, _ = load_sc_dataset(dataset)

        # clustering table processing
        clustering = simplify_configurations(clustering)
        clustering = clustering.loc[~clustering.index.get_level_values('Prior').isin(['StandardNormal'])]
        clustering.reset_index('Trial', drop=True, inplace=True)
        clustering = process_results(clustering, threshold, pm_std=True, no_test_cols=['Utilized clusters'])

        # update clustering table
        clustering['Dataset'] = dataset
        clustering['Donors'] = data.obs['batch'].nunique()
        clustering['Cell types'] = data.obs['cell_type'].nunique()
        new_index = ['Dataset', 'Donors', 'Cell types']
        clustering.set_index(new_index, append=True, inplace=True)
        clustering = clustering.reorder_levels(new_index + ['Prior', 'Config.'])
        clustering_table = pd.concat([clustering_table, clustering])

    # print table
    clustering_table = clustering_table[['Utilized clusters', 'Accuracy', 'NMI (merged)', 'NMI (unmerged)']].copy()
    s = clean_table(clustering_table).style.format(precision=1)
    s.to_latex(os.path.join('results', 'scRNA-clustering.tex'),
               column_format='l' * clustering_table.index.nlevels + '|' + 'c' * len(clustering_table.columns),
               hrules=True,
               clines="skip-last;data",
               multirow_align='t')


def plot_mdes(exp_path):
    # loop over datasets
    for dataset in os.listdir(exp_path):
        try:
            integration = pd.read_pickle(os.path.join(exp_path, dataset, 'testing_integration_performance.pkl'))
            integration = simplify_configurations(integration).sort_index()
        except FileNotFoundError:
            continue
        if len(integration) == 0:
            continue

        # load dataset
        data, _, _, _ = load_sc_dataset(dataset)

        # initialize figure
        models_and_priors = [
            ('scVI', 'StandardNormal'),
            ('scVI', 'GaussianMixture'),
            ('scVI', 'VampPriorMixture'),
        ]
        rows, cols = len(models_and_priors), 3
        size = (5 * cols, 5 * rows + 2)
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=size)

        # loop over models and priors
        for i, (model, prior) in enumerate(models_and_priors):

            # find best trial
            df = integration.loc[pd.IndexSlice[model, prior]]
            assert df.index.get_level_values('Config.').nunique() == 1
            if ('Aggregate score', 'Total') in df.columns:
                config, trial = df[('Aggregate score', 'Total')].idxmax()
            else:
                config, trial = df[('Aggregate score', 'Bio conservation')].idxmax()

            # load embeddings
            model_path = os.path.join(exp_path, dataset, 'trial_{:d}'.format(trial), model, prior, str(config))
            embeddings = np.load(os.path.join(model_path, 'embeddings.npy'))

            # load prior cluster assignments
            if os.path.exists(os.path.join(model_path, 'cluster_probs.npy')):
                cluster_probs = np.load(os.path.join(model_path, 'cluster_probs.npy'))
                cluster_pred = np.argmax(cluster_probs, axis=1)
                _, cluster_pred = np.unique(cluster_pred, return_inverse=True)  # reorder clusters 1, 2, 3, ...
            else:
                cluster_pred = np.zeros_like(data.obs['cell_type'])

            # run mde
            mde = scvi.model.utils.mde(embeddings)
            df = pd.DataFrame(data={
                'x': mde[:, 0],
                'y': mde[:, 1],
                'Batch': data.obs['batch'] + 1,
                'Cell type': data.obs['cell_type'],
                'Cluster': cluster_pred + 1
            })

            # comparison plot
            palette = (
                    sns.color_palette('tab20', n_colors=20)[0::2] +
                    sns.color_palette('tab20', n_colors=20)[1::2]
            )
            kwargs = dict(
                s=3,
                alpha=0.5,
                palette=palette,
                markers=['o' if x < 20 else '^' for x in range(1, 41)],
                rasterized=True,
            )
            legend = 'full' if i == len(models_and_priors) - 1 else False
            sns.scatterplot(df, x='x', y='y', hue='Batch', ax=ax[i, 0], legend=legend, **kwargs)
            sns.scatterplot(df, x='x', y='y', hue='Cell type', ax=ax[i, 1], legend=legend, **kwargs)
            sns.scatterplot(df, x='x', y='y', hue='Cluster', style='Cluster', ax=ax[i, 2], legend=legend, **kwargs)
            for c in range(cols):
                ax[i, c].set_xlabel('')
                ax[i, c].set_xticks([])
                ax[i, c].set_ylabel('')
                ax[i, c].set_yticks([])
                handles, labels = ax[i, c].get_legend_handles_labels()
                ax[i, c].legend(handles=handles, labels=labels, markerscale=5)

            ax[i, 0].set_ylabel('{:s} with {:s} prior'.format(model, clean_prior_name(prior)),
                                fontdict={'fontsize': rcParams['axes.titlesize']})
            if i == 0:
                ax[i, 0].set_title('Batch correction')
                ax[i, 1].set_title('Bio conservation')
                ax[i, 2].set_title('Prior cluster membership')
            if i == len(models_and_priors) - 1:
                sns.move_legend(ax[i, 0], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
                sns.move_legend(ax[i, 1], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
                sns.move_legend(ax[i, 2], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)

        # save figure
        plt.figure(fig)  # set current figure for tight layout after possible cluster assignment plot
        plt.tight_layout()
        fig.savefig(os.path.join('results', 'scRNA-mde-{:s}.pdf'.format(dataset.replace(' ', '-'))), dpi=200)


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()

    # make sure results directory exists
    os.makedirs('results', exist_ok=True)

    # set experiment path and check if it exists
    experiment_path = os.path.join('experiments', 'scRNA-seq', str(args.seed))
    if not os.path.exists(experiment_path):
        exit(code='Experiment directory not found!')

    # tuning plots
    plot_tuning(experiment_path)

    # integration table
    print_integration_table(experiment_path, args.threshold)

    # clustering table
    print_clustering_table(experiment_path, args.threshold)

    # MDE plots
    plot_mdes(experiment_path)

    # show plots
    plt.show()
