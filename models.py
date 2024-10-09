import abc
import argparse
import priors

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import sklearn.mixture as mm

from callbacks import PerformanceMonitor
from datasets import load_tensorflow_dataset
from tensorflow_probability import distributions as tfpd
from tensorflow_probability import layers as tfpl
from tqdm import tqdm
from utils import clustering_metrics, clustering_performance, sample_data_indices

# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def build_encoder(dim_x: list) -> tf.keras.Sequential:
    return tf.keras.Sequential(name='encoder', layers=[
        tf.keras.layers.InputLayer(input_shape=dim_x, dtype=tf.float32),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=500, activation='relu'),
        tf.keras.layers.Dense(units=500, activation='relu'),
        tf.keras.layers.Dense(units=2000, activation='relu'),
    ])


def build_decoder(latent_dim: int, dim_x: list) -> tf.keras.Sequential:
    return tf.keras.Sequential(name='decoder', layers=[
        tf.keras.layers.InputLayer(input_shape=latent_dim, dtype=tf.float32),
        tf.keras.layers.Dense(units=2000, activation='relu'),
        tf.keras.layers.Dense(units=500, activation='relu'),
        tf.keras.layers.Dense(units=500, activation='relu'),
        tf.keras.layers.Dense(units=np.prod(dim_x), activation=None),
        tf.keras.layers.Reshape(target_shape=dim_x),
    ])


class VariationalAutoencoder(tf.keras.Model, abc.ABC):

    def __init__(self, encoder, decoder, prior, **kwargs):
        super().__init__()

        # deep latent variable model
        self.qz = self._define_variational_family(encoder, prior)
        self.decoder = decoder
        self.prior = prior
        self.scale = tfp.util.TransformedVariable(1.0, bijector=tfp.bijectors.Softplus(), name='scale')

        # VI trackers
        self.elbo_tracker = tf.keras.metrics.Mean(name='elbo')
        self.ell_tracker = tf.keras.metrics.Mean(name='ell')
        self.dkl_tracker = tf.keras.metrics.Mean(name='dkl')

        # clustering trackers
        self.cluster_util = tf.keras.metrics.MeanTensor(name='clust')
        self.accuracy_tracker = tf.keras.metrics.Mean(name='acc')
        self.ari_tracker = tf.keras.metrics.Mean(name='ari')
        self.nmi_tracker = tf.keras.metrics.Mean(name='nmi')

    def callbacks(self, **kwargs):
        return list()

    @staticmethod
    def _define_variational_family(encoder, prior):
        encoder.add(tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(prior.latent_dim)))
        encoder.add(tfpl.MultivariateNormalTriL(prior.latent_dim))
        return encoder

    def px(self, z, **kwargs):
        px_z = tfpd.Normal(self.decoder(z, **kwargs), self.scale)
        return tfpd.Independent(px_z, px_z.batch_shape.ndims - 1)

    def variational_objective(self, x, **kwargs):

        # variational objective
        qz_x = self.qz(x, **kwargs)
        px_z = self.px(qz_x.sample(), **kwargs)
        expected_log_likelihood = px_z.log_prob(x)
        kl_divergence = self.prior.kl_divergence(qz_x, encoder=self.qz)
        elbo = expected_log_likelihood - kl_divergence

        # track elbo and its components
        self.elbo_tracker.update_state(elbo)
        self.ell_tracker.update_state(expected_log_likelihood)
        self.dkl_tracker.update_state(kl_divergence)

        return tf.reduce_mean(-elbo), {'qz_x': qz_x}

    def variational_inference_step(self, x, trainable_variables=None):
        trainable_variables = trainable_variables or self.trainable_variables
        with tf.GradientTape() as tape:
            loss, variational_family = self.variational_objective(x, training=True)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, trainable_variables), trainable_variables))

        return variational_family

    def cluster_probabilities(self, z_samples):
        if hasattr(self.prior, 'cluster_probabilities'):
            return self.prior.cluster_probabilities(samples=z_samples, encoder=self.qz)
        else:
            return None

    def additional_metrics(self, data, variational_family):
        cluster_probs = self.cluster_probabilities(variational_family['qz_x'].sample())
        if 'label' in data and cluster_probs is not None:
            cluster_util, accuracy, ari, nmi = \
                clustering_metrics(data['label'], cluster_probs, merge_clusters=False, onehot_clusters=True)
            self.cluster_util.update_state(cluster_util)
            self.accuracy_tracker.update_state(accuracy)
            self.ari_tracker.update_state(ari)
            self.nmi_tracker.update_state(nmi)
        else:
            self.cluster_util.update_state([1])

    def train_step(self, data: dict):
        variational_family = self.variational_inference_step(data['x'])
        self.additional_metrics(data, variational_family)
        return self.get_metrics_result()

    def test_step(self, data: dict):
        _, variational_family = self.variational_objective(data['x'], training=False)
        self.additional_metrics(data, variational_family)
        return self.get_metrics_result()

    def predict_step(self, data: dict):
        _, variational_family = self.variational_objective(data['x'], training=False)
        return variational_family['qz_x'].sample()

    def elbo_surgery(self, data, m=100, batch_size=1000):

        # prior and variational family
        pz = self.prior.pz(encoder=self.qz)

        # loop over the data in batches
        qzn_mu = []  # variational mean
        qzn_sigma = []  # variational covariance
        expected_log_pxn_zn = []  # expected log likelihood w.r.t. q(z|x)
        kl_qzn_pz = []  # KL divergence from ELBO
        log_px = []  # prior predictive log probabilities
        log_px_x = []  # posterior predictive log probabilities
        for xn in tqdm(tf.data.Dataset.from_tensor_slices(data['x']).batch(batch_size)):

            # aggregate local variational parameters
            qzn = self.qz(xn, training=False)
            qzn_mu.append(qzn.mean())
            qzn_sigma.append(qzn.covariance())

            # sample local posterior
            z_samples = qzn.sample(m)

            # expected log likelihood w.r.t. q(z|x)
            pxn_zn = self.px(tf.reshape(z_samples, [-1, self.prior.latent_dim]), training=False)
            xn = tf.tile(xn[None], tf.concat([[m], tf.ones_like(xn.shape)], axis=0))
            log_pxn_zn = pxn_zn.log_prob(tf.reshape(xn, tf.concat([[-1], xn.shape[2:]], axis=0)))
            log_pxn_zn = tf.reshape(log_pxn_zn, [m, batch_size])
            expected_log_pxn_zn.append(tf.reduce_mean(log_pxn_zn, axis=0))

            # KL divergence DKL(q(z|x) || p(z))
            kl_samples = qzn.log_prob(z_samples) - pz.log_prob(z_samples)
            kl_qzn_pz.append(tf.reduce_mean(kl_samples, axis=0))

            # log marginal likelihood via importance weights (Burda et al., 2016)
            w = log_pxn_zn - kl_samples
            log_px.append(tf.math.reduce_logsumexp(w, axis=0) - tf.math.log(float(m)))

            # log posterior predictive
            log_px_x.append(tf.math.reduce_logsumexp(log_pxn_zn, axis=0) - tf.math.log(float(m)))

        # stack variational parameters to create the marginal posterior
        qzn = tfp.distributions.MultivariateNormalTriL(
                loc=tf.concat(qzn_mu, axis=0),
                scale_tril=tf.linalg.cholesky(tf.concat(qzn_sigma, axis=0))
        )
        qz = tfp.distributions.MixtureSameFamily(tfpd.Categorical(logits=tf.zeros(qzn.batch_shape)), qzn)

        # concat and average results
        expected_log_pxn_zn = tf.reduce_mean(tf.concat(expected_log_pxn_zn, axis=0))
        average_kl = tf.reduce_mean(tf.concat(kl_qzn_pz, axis=0))
        log_px = tf.reduce_mean(tf.concat(log_px, axis=0))
        log_px_x = tf.reduce_mean(tf.concat(log_px_x, axis=0))

        # marginal KL
        marginal_kl = tf.zeros([])
        z_samples = tf.transpose(qzn.sample(m), [1, 0, 2])
        for zn in tqdm(tf.data.Dataset.from_tensor_slices(z_samples).batch(1)):
            marginal_kl += tf.reduce_sum(qz.log_prob(zn) - pz.log_prob(zn))
        marginal_kl = marginal_kl / (tf.cast(qzn.batch_shape[0], tf.float32) * m)
        mutual_information = average_kl - marginal_kl

        # pack results
        results = {
            'log p(x)': [log_px.numpy()],
            'log p(x|x)': [log_px_x.numpy()],
            'Distortion': [-expected_log_pxn_zn.numpy()],
            'Avg. KL': [average_kl.numpy()],
            'Marginal KL': [marginal_kl.numpy()],
            'I[z;n]': [mutual_information.numpy()],
        }

        return results


class EmpiricalBayesVariationalAutoencoder(VariationalAutoencoder, abc.ABC):

    def __init__(self, encoder, decoder, prior, **kwargs):
        super().__init__(encoder, decoder, prior)

        # model and prior parameter lists
        self.prior_params = self.prior.trainable_variables
        self.model_params = [p for p in self.trainable_variables if p.name not in [pp.name for pp in self.prior_params]]
        assert len(self.trainable_variables) == len(self.model_params) + len(self.prior_params)

    def train_step(self, data: dict):
        variational_family = self.variational_inference_step(data['x'], trainable_variables=self.model_params)
        self.prior.inference_step(encoder=self.qz, x=data['x'], training=False)
        self.additional_metrics(data, variational_family)
        return self.get_metrics_result()

    def test_step(self, data: dict):
        _, variational_family = self.variational_objective(data['x'], training=False)
        self.additional_metrics(data, variational_family)
        return self.get_metrics_result()


class VariationalDeepEmbedding(VariationalAutoencoder, abc.ABC):
    """Variational deep embedding: An unsupervised and generative approach to clustering (Jiang et al., 2016)"""
    def __init__(self, encoder, decoder, prior, batch_size: int, **kwargs):
        assert isinstance(prior, priors.GaussianMixture)
        assert prior.inference == 'None'
        super().__init__(encoder, decoder, prior)

        self.pretraining = tf.Variable(True, trainable=False, name='pretraining')
        self.pretraining_epochs = round(5 * batch_size / 100)  # same iterations as "a few epochs" w/ batch size of 100
        assert self.pretraining_epochs < 100, 'watch out, early stopping might be triggered'

    def callbacks(self, *, x, **kwargs):
        class Pretraining(tf.keras.callbacks.Callback):

            def __init__(self, pretraining_epochs):
                super().__init__()
                self.pretraining_epochs = pretraining_epochs
                self.x = x
                self.learning_rate = None

            def on_train_begin(self, logs=None):
                self.learning_rate = None

            def on_epoch_begin(self, epoch, logs=None):
                epoch += 1
                if epoch == self.pretraining_epochs:
                    self.model.pretraining.assign(False)
                    gmm = mm.GaussianMixture(n_components=self.model.prior.num_clusters, reg_covar=1e-3)
                    gmm.fit(self.model.qz(self.x).sample())
                    self.model.prior.mu.assign(tf.cast(gmm.means_, self.model.prior.mu.dtype))
                    self.model.prior.L.assign(tf.cast(gmm.precisions_cholesky_, self.model.prior.L.dtype))
                    self.model.prior.pi_logits.assign(tf.cast(np.log(gmm.weights_), self.model.prior.pi_logits.dtype))
                    print('\nGMM initialized')
                    self.model.optimizer.learning_rate = self.learning_rate
                elif self.learning_rate is None:
                    self.learning_rate = self.model.optimizer.learning_rate
                    self.model.optimizer.learning_rate = 2e-3

        return [Pretraining(self.pretraining_epochs)]

    def variational_objective(self, x, **kwargs):

        # priors
        pz_c = self.prior.pz_c()

        # variational objective
        qz_x = self.qz(x, **kwargs)
        z_samples = qz_x.sample()
        qc_x = self.prior.qc(tf.expand_dims(z_samples, axis=1))
        expected_log_likelihood = self.px(z_samples, **kwargs).log_prob(x)
        kl_divergence = qc_x * tf.transpose(tf.vectorized_map(lambda p: qz_x.kl_divergence(p), elems=pz_c))
        kl_divergence += tf.math.xlogy(qc_x, qc_x)
        kl_divergence -= tf.math.multiply_no_nan(tf.nn.log_softmax(self.prior.pi_logits), qc_x)
        kl_divergence = tf.reduce_sum(kl_divergence, axis=1)
        elbo = expected_log_likelihood - tf.where(self.pretraining, 0.0, kl_divergence)

        # track elbo and its components
        self.elbo_tracker.update_state(elbo)
        self.ell_tracker.update_state(expected_log_likelihood)
        self.dkl_tracker.update_state(kl_divergence)

        return tf.reduce_mean(-elbo), {'qz_x': qz_x}


class GaussianMixtureVariationalAutoencoder(tf.keras.Model, abc.ABC):
    """Deep unsupervised clustering with gaussian mixture variational autoencoders (Dilokthanakul et al., 2016)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class KumaraswamyStickBreakingProcess(VariationalAutoencoder):
    """ Kumaraswamy Stick-Breaking Process (Nalisnick & Smyth, 2017)"""
    def __init__(self, encoder, decoder, prior, a_prior=1.0, b_prior=5.0, taylor_order=10, **kwargs):
        super().__init__(encoder, decoder, prior)

        # prior parameters and KL approximation order
        self.a_prior = a_prior
        self.b_prior = b_prior
        self.taylor_order = taylor_order

        # rename qz to encoder since qz just returns the final hidden layer output
        self.encoder = lambda x, **k: self.qz(x, **k)

        # Kumaraswamy encoder parameter heads
        self.a = self._parameter_heads(encoder.output_shape[1], prior.num_clusters, name='a')
        self.b = self._parameter_heads(encoder.output_shape[1], prior.num_clusters, name='b')

    @staticmethod
    def _define_variational_family(encoder, prior):
        return encoder

    @staticmethod
    def _parameter_heads(input_shape, output_shape, name):
        return tf.keras.Sequential(name=name, layers=[
            tf.keras.layers.InputLayer(input_shape),
            tf.keras.layers.Dense(output_shape, activation='softplus')
        ])

    @ staticmethod
    def ksb_sample(a, b):

        # Kumaraswamy inverse CDF sampling
        x = (1 - (1 - tf.random.uniform(tf.shape(a), minval=0.1, maxval=0.99)) ** (1 / b)) ** (1 / a)

        # convert to a Dirichlet sample approximation via Beta string cutting method
        pi = [x[:, 0]]
        for j in range(1, a.shape[1] - 1):
            pi.append((1 - tf.reduce_sum(tf.stack(pi, axis=-1), axis=-1)) * x[:, j])
        pi.append(1 - tf.reduce_sum(tf.stack(pi, axis=-1), axis=-1))
        pi = tf.stack(pi, axis=1)

        return pi

    def ksb_kl_divergence(self, a, b):

        # KL-Divergence
        kl = (a - self.a_prior) / a * (-np.euler_gamma - tf.math.digamma(b) - 1 / b) \
             + (tf.math.log(a * b)) \
             + (tf.math.lbeta(tf.stack((self.a_prior, self.b_prior), axis=-1))) \
             - (b - 1) / b
        for m in range(1, self.taylor_order + 1):
            B = tf.exp(tf.math.lbeta(tf.concat((m / a[..., None], b[..., None]), axis=-1)))
            kl += (self.b_prior - 1) * b / (m + a * b) * B

        # sum over the stick breaking distributions
        kl = tf.reduce_sum(kl, axis=1)

        return kl

    def cluster_probs(self, variational_family):
        return self.ksb_sample(variational_family['a'], variational_family['b'])


class DeepLatentGaussianMixtureModel(KumaraswamyStickBreakingProcess, abc.ABC):
    """Approximate inference for deep latent gaussian mixtures (Nalisnick et al., 2016)"""
    def __init__(self, encoder, decoder, prior, **kwargs):
        assert isinstance(prior, priors.Prior)
        super().__init__(encoder, decoder, prior, **kwargs)

        # component prior
        self.pz = priors.StandardNormal(latent_dim=prior.latent_dim)  # TODO: sample cluster centers

        # encoder heads for Gaussian components
        self.qzk = [tf.keras.Sequential(name='qz_{:d}'.format(i), layers=[
            tf.keras.layers.InputLayer(input_shape=encoder.output_shape[1:], dtype=tf.float32),
            tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(prior.latent_dim)),
            tfpl.MultivariateNormalTriL(prior.latent_dim)
        ]) for i in range(prior.num_clusters)]

    def variational_objective(self, x, **kwargs):

        # variational objective
        enc_out = self.encoder(x, **kwargs)
        a = self.a(enc_out, **kwargs)
        b = self.b(enc_out, **kwargs)
        pi_sample = self.ksb_sample(a, b)
        qzk_x = [qz(enc_out) for qz in self.qzk]
        expected_log_likelihood = tf.stack([self.px(qz_x.sample(), **kwargs).log_prob(x) for qz_x in qzk_x], axis=1)
        expected_log_likelihood = tf.reduce_sum(pi_sample * expected_log_likelihood, axis=1)
        kl_divergence = tf.stack([self.pz.kl_divergence(qz_x) for qz_x in qzk_x], axis=1)
        kl_divergence = tf.reduce_sum(pi_sample * kl_divergence, axis=1)
        kl_divergence += self.ksb_kl_divergence(a, b)
        elbo = expected_log_likelihood - kl_divergence

        # track elbo and its components
        self.elbo_tracker.update_state(elbo)
        self.ell_tracker.update_state(expected_log_likelihood)
        self.dkl_tracker.update_state(kl_divergence)

        return tf.reduce_mean(-elbo), {'a': a, 'b': b}


class StickBreakingVariationalAutoencoder(KumaraswamyStickBreakingProcess, abc.ABC):
    """Stick-Breaking Variational Autoencoders (Nalisnick & Smyth, 2017)"""
    def __init__(self, encoder, decoder, prior, **kwargs):
        assert isinstance(prior, priors.BetaStickBreaking)
        super().__init__(encoder, decoder, prior, **kwargs)

    def variational_objective(self, x, **kwargs):

        # variational objective
        enc_out = self.encoder(x, **kwargs)
        a = self.a(enc_out, **kwargs)
        b = self.b(enc_out, **kwargs)
        expected_log_likelihood = self.px(self.ksb_sample(a, b)).log_prob(x)
        kl_divergence = self.ksb_kl_divergence(a, b)
        elbo = expected_log_likelihood - kl_divergence

        # track elbo and its components
        self.elbo_tracker.update_state(elbo)
        self.ell_tracker.update_state(expected_log_likelihood)
        self.dkl_tracker.update_state(kl_divergence)

        return tf.reduce_mean(-elbo), {'a': a, 'b': b}


def select_model(name: str):
    if name == 'VariationalAutoencoder':
        return VariationalAutoencoder
    elif name == 'EmpiricalBayesVariationalAutoencoder':
        return EmpiricalBayesVariationalAutoencoder
    elif name == 'VariationalDeepEmbedding':
        return VariationalDeepEmbedding
    elif name == 'DeepLatentGaussianMixtureModel':
        return DeepLatentGaussianMixtureModel
    elif name == 'StickBreakingVariationalAutoencoder':
        return StickBreakingVariationalAutoencoder
    else:
        raise NotImplementedError


def clean_model_name(name: str) -> str:
    if name == 'VariationalAutoencoder':
        return 'VAE'
    elif name == 'EmpiricalBayesVariationalAutoencoder':
        return 'EB-VAE'
    elif name == 'VariationalDeepEmbedding':
        return 'VaDE'
    else:
        return name


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--early_stopping_monitor', type=str, default='val_nmi')
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_clusters', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--model', type=str, default='EmpiricalBayesVariationalAutoencoder')
    parser.add_argument('--prior', type=str, default='VampPriorMixture')
    parser.add_argument('--prior_inference', type=str, default='MAP-DP')
    parser.add_argument('--prior_learning_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--vamp_initialization', type=str, default='unlabelled')
    args = parser.parse_args()

    # random seed
    if args.seed is not None:
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(args.seed)

    # load dataset
    x_train, labels_train, x_valid, labels_valid = load_tensorflow_dataset(args.dataset)

    # VampPrior pseudo-input initialization
    assert args.vamp_initialization in {'labelled', 'unlabelled'}
    labels = labels_train if args.vamp_initialization == 'labelled' else None
    u_init = tf.gather(x_train, sample_data_indices(x_train, args.max_clusters, labels))

    # select prior
    latent_prior = priors.select_prior(args.prior, **dict(
        latent_dim=args.latent_dim,
        num_clusters=args.max_clusters,
        u=tf.Variable(u_init, name='u'),
        inference=args.prior_inference,
        learning_rate=args.learning_rate * args.prior_learning_ratio,
    ))

    # construct and compile model
    model = select_model(args.model)(
        encoder=build_encoder(dim_x=x_train.shape.as_list()[1:]),
        decoder=build_decoder(latent_dim=latent_prior.latent_dim, dim_x=x_train.shape.as_list()[1:]),
        prior=latent_prior,
        batch_size=args.batch_size,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), run_eagerly=args.debug)

    # train model
    main_callback = PerformanceMonitor(monitor=args.early_stopping_monitor, patience=100)
    hist = model.fit(x=dict(x=x_train, label=labels_train), validation_data=dict(x=x_valid, label=labels_valid),
                     batch_size=args.batch_size, epochs=args.max_epochs, verbose=False,
                     callbacks=[main_callback] + model.callbacks(x=x_train))

    # clustering performance
    labels = tf.concat([labels_train, labels_valid], axis=0)
    latent_samples = model.predict(dict(x=tf.concat([x_train, x_valid], axis=0)), verbose=False)
    df = clustering_performance(model.cluster_probabilities(latent_samples), labels)
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(df)

    # elbo surgery
    print('\n', model.elbo_surgery(dict(x=x_valid)))
