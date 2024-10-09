import abc
import argparse
import os
import priors
import scvi

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from dataclasses import dataclass
from datasets import load_sc_dataset
from callbacks import PerformanceMonitor
from matplotlib import pyplot as plt
from tensorflow_probability import bijectors as tfpb
from tensorflow_probability import distributions as tfpd
from tensorflow_probability import layers as tfpl
from typing import Literal
from scib_metrics.benchmark import Benchmarker
from typing import Any, Dict, Union
from utils import sample_data_indices

Kwargs = Dict[str, Any]
MetricType = Union[bool, Kwargs]

# configure GPUs for TensorFlow
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def build_network(input_shape, num_layers, hidden_units, dropout, output_shape, name, output_fn=None):
    network = tf.keras.Sequential(
        layers=[tf.keras.layers.InputLayer(input_shape=input_shape, dtype=tf.float32)],
        name=name)
    for _ in range(num_layers):
        network.add(tf.keras.layers.Dense(units=hidden_units))
        network.add(tf.keras.layers.BatchNormalization())
        network.add(tf.keras.layers.ReLU())
        network.add(tf.keras.layers.Dropout(dropout))
    network.add(tf.keras.layers.Dense(units=output_shape, activation=output_fn))
    return network


def dataset_config(dataset):
    if dataset == 'brain large':
        return dict(
            n_layers=3,
            hidden_dim=256,
            latent_dim=10,
            likelihood='zinb',
            learning_rate=1e-3)
    elif dataset == 'cortex':
        return dict(
            n_layers=1,
            hidden_dim=128,
            latent_dim=10,
            likelihood='zinb',
            learning_rate=4e-4)
    elif dataset == 'lung atlas':
        return dict(
            n_layers=2,
            hidden_dim=128,
            latent_dim=30,
            likelihood='nb',
            learning_rate=1e-3)
    elif dataset == 'pbmc':
        return dict(
            n_layers=1,
            hidden_dim=128,
            latent_dim=10,
            likelihood='zinb',
            learning_rate=4e-4)
    elif dataset == 'retina':
        return dict(
            n_layers=1,
            hidden_dim=128,
            latent_dim=10,
            likelihood='zinb',
            learning_rate=5e-4)
    elif dataset == 'split-seq':
        return dict(
            n_layers=2,
            hidden_dim=128,
            latent_dim=30,
            likelihood='zinb',
            learning_rate=1e-3)
    else:
        raise NotImplementedError


class scVI(tf.keras.Model, abc.ABC):

    def __init__(
        self, *,
        n_genes: int,
        n_batches: int,
        prior: priors.Prior,
        hidden_dim: int = 128,
        n_layers: int = 1,
        latent_dim: int = 10,
        latent_cov: Literal['full', 'diag'] = 'diag',
        dropout_rate: float = 0.1,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['zinb', 'nb', 'poisson'] = 'zinb',
        use_observed_library_size: bool = True,
        library_log_loc: tf.constant = None,
        library_log_scale: tf.constant = None,
        **kwargs
    ):
        super().__init__()

        # useful dimensions
        self.dim_x = n_genes
        self.dim_s = n_batches
        self.dim_z = latent_dim

        # priors
        if not use_observed_library_size and (library_log_loc is None or library_log_scale is None):
            raise ValueError('if inferring library size, please provide prior parameters')
        self.pl = None if use_observed_library_size else tfpd.Normal(
            loc=tf.constant(library_log_loc, tf.float32)[:, None],
            scale=tf.constant(library_log_scale, tf.float32)[:, None])
        self.pz = prior

        # latent technical factor variational family
        if self.pl is not None:
            self.encoder_l = build_network(
                input_shape=n_genes,
                num_layers=1,
                hidden_units=hidden_dim,
                dropout=dropout_rate,
                output_shape=2,
                name='encoder_l')

        # latent biological factor variational family
        self.latent_cov = latent_cov
        if self.latent_cov == 'diag':
            output_shape = tfpl.IndependentNormal.params_size(latent_dim)
        elif self.latent_cov == 'full':
            output_shape = tfpl.MultivariateNormalTriL.params_size(latent_dim)
        else:
            raise NotImplementedError
        self.encoder_z = build_network(
            input_shape=n_genes + n_batches,
            num_layers=n_layers,
            hidden_units=hidden_dim,
            dropout=dropout_rate,
            output_shape=output_shape,
            name='encoder_z')

        # gene expression and dropout decoder
        self.decoder = build_network(
            input_shape=latent_dim + n_batches,
            num_layers=n_layers,
            hidden_units=hidden_dim,
            dropout=0.0,  # official scVI implementation's decoder does not use dropout
            output_shape=2 * n_genes,
            name='decoder_w')
        assert dispersion == 'gene'

        # log inverse dispersion
        self.theta = tfp.util.TransformedVariable(tf.ones(n_genes), bijector=tfpb.Exp(), name='theta')

        # likelihood model
        self.likelihood = gene_likelihood

        # model and prior parameter lists
        self.prior_params = self.pz.trainable_variables
        self.model_params = [p for p in self.trainable_variables if p.name not in [pp.name for pp in self.prior_params]]
        assert len(self.trainable_variables) == len(self.model_params) + len(self.prior_params)

        # VI trackers
        self.elbo_tracker = tf.keras.metrics.Mean(name='elbo')
        self.ell_tracker = tf.keras.metrics.Mean(name='ell')
        self.dkl_tracker = tf.keras.metrics.Mean(name='dkl')

        # clustering trackers
        self.cluster_util = tf.keras.metrics.MeanTensor(name='clust')

    def qz(self, xs, **kwargs):
        log_pseudo_counts = tf.math.log1p(tf.cast(xs[0], tf.float32))
        params_z = self.encoder_z(tf.concat([log_pseudo_counts, xs[1]], axis=1), **kwargs)
        if self.latent_cov == 'diag':
            return tfpd.MultivariateNormalDiag(
                loc=params_z[..., :self.dim_z],
                scale_diag=tf.nn.softplus(params_z[..., self.dim_z:])
            )
        elif self.latent_cov == 'full':
            return tfpd.MultivariateNormalTriL(
                loc=params_z[..., :self.dim_z],
                scale_tril=tfp.bijectors.FillScaleTriL()(params_z[..., self.dim_z:]))
        else:
            raise NotImplementedError

    def ql(self, x, **kwargs):
        params_l = self.encoder_l(tf.cast(x, tf.float32), **kwargs)
        return tfpd.Normal(
            loc=params_l[..., 0],
            scale=1e-5 + tf.nn.softplus(params_l[..., 1]))

    def log_px_zls(self, x, l, fw, fh, theta, eps=1e-8):
        x = tf.cast(x, tf.float32)
        mu = l * fw
        log_p = tf.math.log(mu + eps) - tf.math.log(mu + theta + eps)
        log_1_minus_p = tf.math.log(theta + eps) - tf.math.log(mu + theta + eps)
        if self.likelihood == 'nb':
            log_prob = x * log_p + theta * log_1_minus_p + \
                tf.math.lgamma(x + theta) - tf.math.lgamma(theta) - tf.math.lgamma(x + 1)
        elif self.likelihood == 'zinb':
            zero_ll = tf.nn.softplus(-fh + theta * log_1_minus_p) - tf.nn.softplus(-fh)
            non_zero_ll = -fh - tf.nn.softplus(-fh) + theta * log_1_minus_p + x * log_p + \
                tf.math.lgamma(x + theta) - tf.math.lgamma(theta) - tf.math.lgamma(x + 1)
            log_prob = tf.where(tf.less_equal(x, eps), zero_ll, non_zero_ll)
        else:
            raise NotImplementedError

        return tf.reduce_sum(log_prob, axis=-1)

    def variational_objective(self, inputs, **kwargs):

        # amortized variational family and its MC samples
        qz_xs = self.qz((inputs['x'], inputs['s']), **kwargs)
        z_samples = qz_xs.sample()
        if self.pl is None:
            ql_x = None
            l_samples = tf.reduce_sum(tf.cast(inputs['x'], tf.float32), axis=1, keepdims=True)
        else:
            ql_x = self.ql(inputs['x'], **kwargs)
            l_samples = tf.exp(ql_x.sample()[:, None])

        # likelihood parameter maps
        encoder_out = self.decoder(tf.concat([z_samples, inputs['s']], axis=1), **kwargs)
        fw = tf.nn.softmax(encoder_out[..., :self.dim_x], axis=-1)
        fh = encoder_out[..., self.dim_x:]

        # evidence lower bound
        expected_log_likelihood = self.log_px_zls(inputs['x'], l_samples, fw, fh, self.theta)
        kl_divergence = self.pz.kl_divergence(qz_xs, encoder=self.qz)
        if self.pl is not None:
            kl_divergence += tf.reduce_sum(tf.transpose(ql_x.kl_divergence(self.pl)) * inputs['s'], axis=1)
        elbo = expected_log_likelihood - kl_divergence

        # track elbo and its components
        self.elbo_tracker.update_state(elbo)
        self.ell_tracker.update_state(expected_log_likelihood)
        self.dkl_tracker.update_state(kl_divergence)

        return tf.reduce_mean(-elbo), {'qz_xs': qz_xs}

    def cluster_probabilities(self, z_samples):
        if hasattr(self.pz, 'cluster_probabilities'):
            return self.pz.cluster_probabilities(samples=z_samples, encoder=self.qz)
        else:
            return tf.ones_like(z_samples)[:, :1]

    def additional_metrics(self, inputs, variational_family):
        cluster_probs = self.cluster_probabilities(variational_family['qz_xs'].sample())
        if cluster_probs is not None:
            predicted_labels = tf.argmax(cluster_probs, axis=1)
            cluster_util = tf.reduce_max(tf.one_hot(predicted_labels, depth=tf.shape(cluster_probs)[1]), axis=0)
            self.cluster_util.update_state(cluster_util)
        else:
            self.cluster_util.update_state([1])

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss, variational_family = self.variational_objective(inputs, training=True)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.model_params), self.model_params))
        self.pz.inference_step(encoder=self.qz, x=(inputs['x'], inputs['s']), training=False)
        self.additional_metrics(inputs, variational_family)

        return self.get_metrics_result()

    def test_step(self, inputs):
        _, variational_family = self.variational_objective(inputs, training=False)
        self.additional_metrics(inputs, variational_family)
        return self.get_metrics_result()

    def predict_step(self, inputs: dict):
        _, variational_family = self.variational_objective(inputs, training=False)
        return variational_family['qz_xs'].mean()


def vamp_prior_pseudo_inputs(count_matrix, one_hot_batch_id, num_clusters, cell_labels=None):
    idx = sample_data_indices(count_matrix, num_clusters, cell_labels)
    count_matrix = tf.cast(count_matrix, tf.float32)
    one_hot_batch_id = tf.nn.softmax(tf.zeros_like(one_hot_batch_id))
    chain = tfpb.Chain([tfpb.Shift(-1), tfpb.Softplus()])
    u_counts = tfp.util.TransformedVariable(tf.gather(count_matrix, idx), chain, name='u_counts')
    u_batch = tfp.util.TransformedVariable(tf.gather(one_hot_batch_id, idx), tfpb.SoftmaxCentered(), name='u_batch')
    return u_counts, u_batch


def plot_mde_of_embeddings(mde, batch, cell_type, cluster=None, fig_path=None):
    plot_cluster_assignments = cluster is not None
    n_cols = 2 + plot_cluster_assignments
    fig, ax = plt.subplots(ncols=n_cols, figsize=(5 * n_cols, 5), subplot_kw=dict(xticks=tuple(), yticks=tuple()))
    kwargs = dict(s=3, alpha=0.5, palette='tab20')
    sns.scatterplot(x=mde[:, 0], y=mde[:, 1], hue=(1 + batch).astype(str), ax=ax[0], **kwargs)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, title='Technical batch')
    sns.scatterplot(x=mde[:, 0], y=mde[:, 1], hue=cell_type, hue_order=np.unique(cell_type), ax=ax[1], **kwargs)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, title='Cell type')
    if plot_cluster_assignments:
        sns.scatterplot(x=mde[:, 0], y=mde[:, 1], hue=(1 + cluster).astype(str), ax=ax[2], **kwargs)
        ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4, title='Cluster assignment')
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)


@dataclass(frozen=True)
class NoBatchCorrection:
    """
    A class to skip batch correction metrics that won't work if only a single batch exists
    """
    silhouette_batch: MetricType = False
    ilisi_knn: MetricType = False
    kbet_per_label: MetricType = False
    graph_connectivity: MetricType = False
    pcr_comparison: MetricType = True


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--dataset', type=str, default='pbmc')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--max_clusters', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--prior', type=str, default='VampPriorMixture')
    parser.add_argument('--prior_inference', type=str, default='MAP-DP')
    parser.add_argument('--prior_learning_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--vamp_initialization', type=str, default='labelled')
    args = parser.parse_args()

    # save directory
    save_dir = os.path.join('results', 'scVI-runs', args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # load data and get model configuration
    data, counts, batch_id, labels = load_sc_dataset(args.dataset)
    model_config = dataset_config(args.dataset)

    # log library sizes conditioned on batch
    log_counts_batch = np.ma.log(tf.einsum('ij,ik->ik', tf.cast(counts, tf.float32), batch_id))

    # initialize baseline prior and proposed prior
    baseline_prior = priors.StandardNormal(latent_dim=model_config['latent_dim'])
    assert args.vamp_initialization in {'labelled', 'unlabelled'}
    labels = labels if args.vamp_initialization == 'labelled' else None
    proposed_prior = priors.select_prior(args.prior, **dict(
        latent_dim=model_config['latent_dim'],
        num_clusters=args.max_clusters,
        u=vamp_prior_pseudo_inputs(counts, batch_id, args.max_clusters, labels),
        inference=args.prior_inference,
        learning_rate=model_config['learning_rate'] * args.prior_learning_ratio,
    ))

    # loop over the two priors
    for latent_prior in [proposed_prior, baseline_prior]:

        # set random seed
        if args.seed is not None:
            tf.config.experimental.enable_op_determinism()
            tf.keras.utils.set_random_seed(args.seed)

        # construct and compile model
        model = scVI(
            n_genes=counts.shape[1],
            n_batches=batch_id.shape[1],
            prior=latent_prior,
            use_observed_library_size=True,
            library_log_loc=np.mean(log_counts_batch, axis=0),
            library_log_scale=np.std(log_counts_batch, axis=0),
            **model_config
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(model_config['learning_rate']), run_eagerly=args.debug)
        model.optimizer.build(model.trainable_variables)

        # train model
        i_train = np.random.choice(counts.shape[0], size=int(counts.shape[0] * args.train_ratio), replace=False).tolist()
        i_valid = list(set(range(counts.shape[0])) - set(i_train))
        data_train = dict(x=tf.gather(counts, i_train), s=tf.gather(batch_id, i_train), c=tf.gather(labels, i_train))
        data_valid = dict(x=tf.gather(counts, i_valid), s=tf.gather(batch_id, i_valid), c=tf.gather(labels, i_valid))
        hist = model.fit(
            x=data_train,
            validation_data=data_valid,
            batch_size=args.batch_size,
            epochs=args.max_epochs,
            verbose=False,
            callbacks=[PerformanceMonitor(patience=args.patience)]
        )

        # get latent representation and cluster assignments
        short_name = 'scVI_' + ''.join([c for c in latent_prior.name if c.isupper()])
        data.obsm[short_name] = model.predict(dict(x=counts, s=batch_id), batch_size=args.batch_size)
        data.obs['cluster'] = np.argmax(model.cluster_probabilities(data.obsm[short_name]).numpy(), axis=1)

        # plot MDE of latent embeddings
        if args.dataset != 'brain large':
            fig_name = short_name + '-{:d}-{:.2f}.pdf'.format(args.batch_size, args.prior_learning_ratio)
            plot_mde_of_embeddings(
                mde=scvi.model.utils.mde(data.obsm[short_name]),
                batch=data.obs['batch'],
                cell_type=data.obs['cell_type'],
                cluster=data.obs['cluster'],
                fig_path=os.path.join(save_dir, fig_name))

    # benchmark performance
    bm = Benchmarker(
        data,
        batch_key='batch',
        label_key='cell_type',
        embedding_obsm_keys=[k for k in data.obsm.keys() if 'scVI' in k],
        batch_correction_metrics=NoBatchCorrection() if data.obs['batch'].nunique() == 1 else None,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, save_dir=save_dir)
    os.rename(os.path.join(save_dir, 'scib_results.svg'),
              os.path.join(save_dir, '{:d}-{:.2f}.svg'.format(args.batch_size, args.prior_learning_ratio)))
