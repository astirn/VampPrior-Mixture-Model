import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import ttest_rel
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


def sample_data_indices(data, num_samples, labels=None):
    # if no class labels were provided, randomly select data
    if labels is None:
        return tf.random.shuffle(tf.range(tf.shape(data, out_type=tf.int64)[0]))[:num_samples]

    # if class labels were provided, randomly select an equal amount of data from each class
    samples_per_label = num_samples // len(tf.unique(labels)[0])
    assert samples_per_label > 0
    indices = []
    for label in tf.unique(labels)[0]:
        indices += [tf.random.shuffle(tf.squeeze(tf.where(tf.equal(label, labels))))[:samples_per_label]]

    # sample remainder of requested samples from population
    remaining_samples = num_samples % len(tf.unique(labels)[0])
    indices += [tf.random.shuffle(tf.range(tf.shape(data, out_type=tf.int64)[0]))[:remaining_samples]]

    return tf.concat(indices, axis=0)


@tf.function
def tf_ari(true_labels, predicted_labels):
    ari = tf.numpy_function(func=adjusted_mutual_info_score, inp=[true_labels, predicted_labels], Tout=tf.float64)
    return tf.cast(ari, tf.float32)


@tf.function
def tf_nmi(true_labels, predicted_labels):
    nmi = tf.numpy_function(func=normalized_mutual_info_score, inp=[true_labels, predicted_labels], Tout=tf.float64)
    return tf.cast(nmi, tf.float32)


def clustering_metrics(labels, cluster_probs, merge_clusters=True, onehot_clusters=False):
    if cluster_probs is None:
        return None, None, None, None

    # ensure true labels are integers
    _, true_labels = tf.unique(labels)

    # predicted labels and the number of utilized clusters
    predicted_labels = tf.argmax(cluster_probs, axis=1)
    num_clusters = tf.squeeze(tf.shape(tf.unique(predicted_labels)[0]))
    cluster_util = tf.reduce_max(tf.one_hot(predicted_labels, depth=tf.shape(cluster_probs)[1]), axis=0)

    # ARI and NMI for unmerged cluster predictions (will be overwritten if merge_clusters is true)
    ari = tf_ari(true_labels, predicted_labels)
    nmi = tf_nmi(true_labels, predicted_labels)

    # match a cluster's label to the true label of the datum with the highest probability of belonging to the cluster
    cluster_labels = tf.gather(true_labels, tf.argmax(cluster_probs, axis=0))
    predicted_labels = tf.gather(cluster_labels, tf.argmax(cluster_probs, axis=1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(true_labels, predicted_labels), tf.float32))

    # ARI and NMI for merged cluster predictions
    if merge_clusters:
        ari = tf_ari(true_labels, predicted_labels)
        nmi = tf_nmi(true_labels, predicted_labels)

    return (cluster_util if onehot_clusters else num_clusters), accuracy, ari, nmi


def clustering_performance(cluster_probs, labels, index=None):
    clusters, acc, ari_merged, nmi_merged = clustering_metrics(labels, cluster_probs, merge_clusters=True)
    _, _, ari_unmerged, nmi_unmerged = clustering_metrics(labels, cluster_probs, merge_clusters=False)
    return pd.DataFrame(
        data={'Accuracy': [acc.numpy() if acc is not None else np.nan],
              'ARI (merged)': [ari_merged.numpy() if acc is not None else np.nan],
              'ARI (unmerged)': [ari_unmerged.numpy() if acc is not None else np.nan],
              'NMI (merged)': [nmi_merged.numpy() if acc is not None else np.nan],
              'NMI (unmerged)': [nmi_unmerged.numpy() if acc is not None else np.nan],
              'Utilized clusters': [clusters.numpy() if clusters is not None else np.nan]},
        index=index if index is not None else [None])


@tf.function
def tf_multi_log_gamma(a, p):
    lg = tf.reduce_sum(tf.stack([tf.math.lgamma(a + (i - 1) / 2) for i in range(1, p + 1)]), axis=0)
    lg += p * (p - 1) / 4 * tf.math.log(np.pi)
    return lg


@tf.function
def tf_multi_digamma(a, p):
    return tf.reduce_sum(tf.stack([tf.math.digamma(a + (i - 1) / 2) for i in range(1, p + 1)]), axis=0)


@tf.function
def wishart_kld(n0, L0, n1, L1):
    p = L0.shape[-1]
    V0 = L0 @ tf.transpose(L0, [0, 2, 1])
    V1_inv = tf.linalg.cholesky_solve(L1, tf.eye(p))
    return (-n1 / 2 * tf.linalg.slogdet(V1_inv @ V0)[1] +
            n0 / 2 * (tf.linalg.trace(V1_inv @ V0) - tf.cast(p, tf.float32)) +
            tf_multi_log_gamma(n1 / 2, p) - tf_multi_log_gamma(n0 / 2, p) +
            (n0 - n1) / 2 * tf_multi_digamma(n0 / 2, p))


def process_results(df, threshold, mode='max', pm_std=False, no_test_cols=None):
    mean = df.groupby(level=df.index.names, sort=False).mean()
    std = df.groupby(level=df.index.names, sort=False).std()
    df.sort_index(inplace=True)
    for col in mean.columns:
        best = mean[col].idxmax() if mode == 'max' else mean[col].idxmin()
        for i in mean.index:
            if pm_std and mean.loc[i, col] > 1:
                mean.loc[i, col] = '{:.3g} $\\pm$ {:.2f}'.format(mean.loc[i, col], std.loc[i, col])
            elif pm_std:
                mean.loc[i, col] = '{:.3f} $\\pm$ {:.2f}'.format(mean.loc[i, col], std.loc[i, col])
            else:
                mean.loc[i, col] = '{:.4f}'.format(mean.loc[i, col])
            p = ttest_rel(df.loc[best, col], df.loc[i, col]).pvalue if i != best else None
            if (i == best or p >= threshold) and (no_test_cols is None or col not in no_test_cols):
                mean.loc[i, col] = '\\textbf{' + mean.loc[i, col] + '}'

    return mean
