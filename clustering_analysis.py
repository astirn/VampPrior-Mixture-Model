import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from datasets import load_tensorflow_dataset
from elbo_surgery_analysis import clean_table
from priors import select_prior, clean_prior_name
from matplotlib import rcParams
from models import select_model, build_encoder, build_decoder
from utils import sample_data_indices, process_results


# avoid type 3 fonts
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


def plot_tuning(exp_path):

    # accumulate tuning values
    tuning = pd.DataFrame()
    for dataset in os.listdir(exp_path):
        dataset_path = os.path.join(exp_path, dataset)
        for dim_z in os.listdir(dataset_path):
            try:
                df = pd.read_pickle(os.path.join(dataset_path, dim_z, 'tuning_performance.pkl'))
                df['Dataset'] = dataset.replace('_', ' ')
                tuning = pd.concat([tuning, df])
            except FileNotFoundError:
                continue

    if len(tuning) == 0:
        return

    # pop and rename relevant parameters
    tuning.reset_index('save_path', drop=True, inplace=True)
    parameters = dict(
        batch_size='Batch size',
        learning_rate='Learning rate',
        prior_learning_ratio='Prior learning ratio',
        num_clusters='Max clusters')
    tuning.reset_index(list(parameters.keys()), inplace=True)
    tuning.rename(columns=parameters, inplace=True)

    # make sure assumptions hold
    assert tuning.index.nunique() == 1, "doesn't support multiple (models, priors, inference)"
    assert tuning['Learning rate'].nunique() == 1, "only support for 1 learning rate"

    # plot results
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    kwargs = dict(
        hue='Dataset' if tuning['Dataset'].nunique() > 1 else None,
        style='Prior learning ratio' if tuning['Prior learning ratio'].nunique() > 1 else None,
        marker='o')
    sns.lineplot(tuning, x='Batch size', y='Utilized clusters', ax=ax[0], **kwargs)
    sns.move_legend(ax[0], loc='upper left', bbox_to_anchor=(0.0, 1.0))
    tuning.rename(columns={'NMI (unmerged)': 'NMI'}, inplace=True)
    sns.lineplot(tuning, x='Batch size', y='NMI', legend=False, ax=ax[1], **kwargs)
    ax[0].set_xticks(tuning['Batch size'].unique())
    ax[1].set_xticks(tuning['Batch size'].unique())
    offset = 0.0175 * (max(ax[1].get_yticks()) - min(ax[1].get_yticks()))
    miny, maxy = min(ax[1].get_yticks()) + offset, max(ax[1].get_yticks()) - offset
    ax[1].plot(
        [236, 276, 276, 236, 236],
        2 * [miny] + 2 * [maxy] + [miny],
        color='tab:green',
        linestyle=':')
    plt.tight_layout()
    fig.savefig(os.path.join('results', 'clustering-tuning.pdf'))


def print_clustering_table(exp_path, threshold):
    # columns of interest
    columns = ['Utilized clusters', 'Accuracy', 'NMI (merged)', 'NMI (unmerged)']

    # loop over datasets
    clustering_table = pd.DataFrame()
    for dataset in os.listdir(exp_path):
        dataset_path = os.path.join(exp_path, dataset)
        for dim_z in os.listdir(dataset_path):
            try:
                clustering = pd.read_pickle(os.path.join(dataset_path, dim_z, 'testing_performance.pkl'))
            except FileNotFoundError:
                continue
            if len(clustering) == 0:
                continue

            # rename columns
            clustering.reset_index(inplace=True)
            clustering.rename(columns=dict(
                model='Model',
                prior='Prior',
            ), inplace=True)

            # dataset name
            clustering['Dataset'] = dataset.replace('_', ' ') + ' w/ $\\' + dim_z.replace('dim_', 'dim(z) = ') + '$'

            # reindex and filter columns
            clustering.fillna('--', inplace=True)  # Inference is NaN for models that don't have it
            clustering.set_index(['Dataset', 'Model', 'Prior'], inplace=True)
            clustering = clustering.loc[~clustering['inference'].isin(['MAP']), columns].copy()

            # clustering table processing
            clustering = process_results(clustering, threshold, pm_std=True, no_test_cols=['Utilized clusters'])
            clustering_table = pd.concat([clustering_table, clustering])

    # print table
    index_order = clustering_table.index.droplevel('Dataset').unique()
    clustering_table = clustering_table.unstack('Dataset').loc[index_order]
    clustering_table.columns = clustering_table.columns.reorder_levels([1, 0])
    datasets = clustering_table.columns.get_level_values(0).unique().to_list()[::-1]
    clustering_table = clean_table(clustering_table[datasets].copy())
    s = clustering_table.style
    col_fmt = 'l' * clustering_table.index.nlevels + ''.join(['|' + 'r' * len(columns) for _ in datasets])
    s.to_latex(os.path.join('results', 'clustering-performance.tex'),
               column_format=col_fmt,
               multicol_align='c|',
               hrules=True,
               multirow_align='t')


def close_factors(number):
    factor1 = 0
    factor2 = number
    while factor1 + 1 <= factor2:
        factor1 += 1
        if number % factor1 == 0:
            factor2 = number // factor1

    return factor1, factor2


def num_rows_and_cols(number):
    while True:
        rows, cols = close_factors(number)
        if rows / 2 <= cols:
            break
        number += 1
    return rows, cols


def plot_generated_data(exp_path, mixture_probs, seed):
    assert mixture_probs in {'prior', 'posterior'}

    # loop over experiments
    for dataset in os.listdir(exp_path):
        dataset_path = os.path.join(exp_path, dataset)
        for dim_z in os.listdir(dataset_path):
            try:
                df = pd.read_pickle(os.path.join(dataset_path, dim_z, 'testing_performance.pkl'))
            except FileNotFoundError:
                continue
            if len(df) == 0:
                continue

            # load dataset
            x_train, labels_train, x_valid, labels_valid = load_tensorflow_dataset(dataset)

            # reset index
            df.reset_index(drop=False, inplace=True)

            # loop over priors of interest
            for p, i in [('VampPrior', None), ('VampPriorMixture', 'MAP-DP')]:

                # find model's best run
                best_run = df.loc[(df.prior == p) & (df.inference.isna() if i is None else df.inference == i)]
                best_run = best_run.iloc[best_run['NMI (unmerged)'].argmax()]
                p = clean_prior_name(p)

                # load model
                prior = best_run['prior']
                del best_run['prior']
                prior = select_prior(prior, latent_dim=int(dim_z.replace('dim_', '')), **best_run.to_dict(), **dict(
                    u=tf.Variable(tf.gather(x_train, sample_data_indices(x_train, best_run['num_clusters'])), name='u'),
                ))
                model = select_model(best_run['model'])(
                    encoder=build_encoder(dim_x=x_train.shape.as_list()[1:]),
                    decoder=build_decoder(latent_dim=int(dim_z.replace('dim_', '')), dim_x=x_train.shape.as_list()[1:]),
                    prior=prior,
                    **best_run.to_dict(),
                )
                checkpoint = tf.train.Checkpoint(model)
                checkpoint.restore(os.path.join(best_run['save_path'], 'best_checkpoint')).expect_partial()

                # utilized clusters' prior probabilities
                tf.keras.utils.set_random_seed(seed)
                cluster_probs = model.cluster_probabilities(model.predict(dict(x=x_train), verbose=False))
                utilized_clusters = tf.unique(tf.argmax(cluster_probs, axis=1))[0]
                if mixture_probs == 'prior':
                    if hasattr(model.prior, 'pi_logits'):
                        prior_probs = tf.nn.softmax(model.prior.pi_logits)
                    else:
                        prior_probs = tf.ones(cluster_probs.shape[0]) / cluster_probs.shape[1]
                elif mixture_probs == 'posterior':
                    prior_probs = tf.reduce_sum(cluster_probs, axis=0) / tf.reduce_sum(cluster_probs)
                else:
                    raise NotImplementedError
                prior_probs = tf.gather(prior_probs, utilized_clusters)

                # sort utilized clusters by prior probabilities
                i_sort = tf.argsort(-prior_probs)
                utilized_clusters = tf.gather(utilized_clusters, i_sort).numpy()
                prior_probs = tf.gather(prior_probs, i_sort).numpy()

                # initialize figure
                max_samples = 10
                rows = max_samples + 1
                fig = plt.figure(figsize=(len(utilized_clusters), rows), constrained_layout=True)
                font_dict = {'size': 11}
                gs = fig.add_gridspec(rows, len(utilized_clusters))

                # loop over clusters
                for j, cluster in enumerate(utilized_clusters):
                    # x-axis label
                    x_label = '$\\pi_{{{:d}}} = {:.2f}\\%$'.format(j + 1, 100 * prior_probs[j])

                    # plot most-probable member
                    ax = fig.add_subplot(gs[-1, j])
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.imshow(x_train[tf.argmax(cluster_probs[:, cluster])], cmap='Greys')
                    ax.set_xlabel(x_label, fontdict=font_dict)
                    if j == 0:
                        ax.set_ylabel('Most prob. $x$', fontdict=font_dict)

                    # sample prior component
                    num_samples = round(prior_probs[j] / max(prior_probs) * max_samples)
                    z_samples = model.prior.pz_c(encoder=model.qz).sample(max_samples)
                    for k in range(num_samples):
                        ax = fig.add_subplot(gs[max_samples - k - 1, j])
                        ax.set_xticks([]), ax.set_yticks([])
                        ax.imshow(tf.squeeze(model.px(z_samples[k, cluster][None]).sample()), cmap='Greys')
                        if j == 0 and k == 0:
                            ax.set_ylabel('   Samples $\longrightarrow$', fontdict=font_dict)
                # save figure
                fig.savefig(os.path.join('results', 'generated-{:s}-{:s}-{:s}.pdf'.format(dataset, p, mixture_probs)))


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()

    # make sure results directory exists
    os.makedirs('results', exist_ok=True)

    # base bath
    experiment_path = os.path.join('experiments', 'clustering', str(args.seed))
    if not os.path.exists(experiment_path):
        exit(code='Experiment directory not found!')

    # tuning plots
    plot_tuning(experiment_path)

    # # clustering table
    # print_clustering_table(experiment_path, args.threshold)
    #
    # # plot generated data
    # plot_generated_data(experiment_path, mixture_probs='prior', seed=args.seed)
    # plot_generated_data(experiment_path, mixture_probs='posterior', seed=args.seed)

    # show plots
    plt.show()
