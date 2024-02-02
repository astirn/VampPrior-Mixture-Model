import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import bijectors as tfpb
from tensorflow_probability import distributions as tfpd
from typing import Tuple, Union

U = Union[tf.Variable, tfp.util.TransformedVariable]


class Prior(tf.keras.layers.Layer):
    def __init__(self, *, latent_dim: int, num_clusters: int, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters

    def inference_step(self, **kwargs):
        pass

    def kl_divergence(self, qz_x: tfpd.Distribution, **kwargs):
        raise NotImplementedError


class StandardNormal(Prior):
    def __init__(self, *, latent_dim: int, **kwargs):
        super().__init__(latent_dim=latent_dim, num_clusters=kwargs.get('num_clusters') or 1)
        self._pz = tfpd.MultivariateNormalDiag(loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim))

    @property
    def name(self):
        return 'StandardNormal'

    def pz(self, **kwargs) -> tfpd.Distribution:
        return self._pz

    def kl_divergence(self, qz_x: tfpd.Distribution, **kwargs) -> tf.Tensor:
        return qz_x.kl_divergence(self._pz)


class BetaStickBreaking(Prior):
    def __init__(self, *, num_clusters: int, **kwargs):
        super().__init__(latent_dim=num_clusters, num_clusters=num_clusters)


class ClusteringPrior(Prior):
    def __init__(
            self, *,
            latent_dim: int,
            num_clusters: int,
            inference: str,
            learning_rate: float,
            pseudo_batch_size: int = 1,
            **kwargs):
        super().__init__(latent_dim=latent_dim, num_clusters=num_clusters)

        # inference
        assert inference in {None, 'None', 'MLE', 'MAP', 'MAP-DP'}
        self.inference = str(inference)
        if self.inference != 'None':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.pseudo_batch_size = tf.constant(pseudo_batch_size, tf.float32)

        # priors
        self.prior_alpha = tfpd.Gamma(1.0, 1.0)
        self.prior_mu = tfpd.MultivariateNormalTriL(loc=tf.zeros(latent_dim), scale_tril=tf.eye(latent_dim))
        self.prior_L = tfpd.WishartTriL(
            df=latent_dim + 2,
            # scale_tril=tf.eye(latent_dim) * num_clusters ** (1 / latent_dim),
            # scale_tril=tf.eye(latent_dim) / (latent_dim + 2) ** 0.5,
            scale_tril=tf.eye(latent_dim) * num_clusters ** (1 / latent_dim) / (latent_dim + 2) ** 0.5,
            input_output_cholesky=True
        )

        # mixture parameters
        self.alpha = tfp.util.TransformedVariable(
            initial_value=1.0,
            bijector=tfpb.Softplus(),
            name='alpha', trainable='DP' in self.inference)
        self.pi_logits = tf.Variable(
            initial_value=tf.zeros(num_clusters),
            name='pi_logits')

        # component precisions
        self.L = tfp.util.TransformedVariable(
            initial_value=tf.eye(latent_dim, batch_shape=[num_clusters]),
            bijector=tfpb.FillScaleTriL(),
            name='L')

        # empirical Bayes trackers
        self.prior_loss = tf.metrics.Mean('prior')

    def pz(self, **kwargs) -> tfpd.Distribution:
        return tfp.distributions.MixtureSameFamily(tfpd.Categorical(logits=self.pi_logits), self.pz_c(**kwargs))

    def kl_divergence(self, qz_x: tfpd.Distribution, **kwargs) -> tf.Tensor:
        return -qz_x.entropy() - self.pz(**kwargs).log_prob(qz_x.sample())

    def cluster_probabilities(self, *, samples, **kwargs) -> tf.Tensor:
        return self.qc(tf.expand_dims(samples, axis=1), **kwargs)

    def log_prior_prob_pi(self) -> tf.Tensor:
        alpha = tf.ones(self.num_clusters) / self.num_clusters / self.alpha
        return tf.reduce_sum((alpha - 1.0) * tf.nn.log_softmax(self.pi_logits), axis=-1) - tf.math.lbeta(alpha)

    def log_prior_prob_mu(self, **kwargs) -> tf.Tensor:
        raise NotImplementedError

    def log_prob_z_c(self, z: tf.Tensor, **kwargs) -> tf.Tensor:
        raise NotImplementedError

    def qc(self, z, **kwargs) -> tf.Tensor:
        return tf.nn.softmax(self.log_prob_z_c(z, **kwargs) + tf.nn.log_softmax(self.pi_logits))

    def inference_step(self, *, q: callable, x: tf.Tensor, **kwargs):

        # sample data from variational posterior
        z = tf.expand_dims(q(x, **kwargs).sample(), axis=1)
        n = tf.cast(tf.shape(z)[0], tf.float32)

        # E-step
        with tf.GradientTape() as tape:
            log_prob_z_c = self.log_prob_z_c(z, encoder=q, **kwargs)
            log_prob_c = tf.nn.log_softmax(self.pi_logits)
            qc_z = tf.nn.softmax(log_prob_z_c + log_prob_c)
            loss = -tf.reduce_sum(qc_z * (log_prob_z_c + log_prob_c)) / n
            if self.inference == 'MAP-DP':
                loss -= self.prior_alpha.log_prob(self.alpha) / n
            if 'MAP' in self.inference:
                loss -= self.log_prior_prob_pi() / n
                loss -= tf.reduce_sum(self.log_prior_prob_mu(encoder=q, **kwargs)) / n
                loss -= tf.reduce_sum(self.prior_L.log_prob(self.L)) / n

        # M-step
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        # loss monitoring
        self.prior_loss.update_state(-loss)

        # NaN/Inf assertions for debugging
        for var in self.trainable_variables:
            tf.assert_equal(tf.math.logical_or(tf.math.is_inf(var), tf.math.is_nan(var)), False, message=var.name)


class GaussianMixture(ClusteringPrior):
    def __init__(self, *, latent_dim: int, num_clusters: int, **kwargs):
        super().__init__(latent_dim=latent_dim, num_clusters=num_clusters, **kwargs)
        self.mu = tf.Variable(tf.random.normal([num_clusters, latent_dim]), name='mu')

    @property
    def name(self):
        return 'GaussianMixture'

    def log_prior_prob_mu(self, **kwargs) -> tf.Tensor:
        return self.prior_mu.log_prob(self.mu)

    def pz_c(self, **kwargs) -> tfpd.Distribution:
        L = tf.linalg.cholesky(tf.linalg.cholesky_solve(self.L, tf.eye(self.latent_dim)))
        return tfpd.MultivariateNormalTriL(self.mu, L)

    def log_prob_z_c(self, z: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.pz_c(**kwargs).log_prob(z)


class VampPriorMixture(ClusteringPrior):
    def __init__(self, *, latent_dim: int, num_clusters: int, u: Union[U, Tuple[U]], **kwargs):
        super().__init__(latent_dim=latent_dim, num_clusters=num_clusters, **kwargs)
        self.u = u
        tf.assert_equal(self.prior_mu.mean(), tf.zeros(self.prior_mu.event_shape),
                        message='log_prior_prob_mu class method assumes this fact is true')
        tf.assert_equal(self.prior_mu.covariance(), tf.eye(self.prior_mu.event_shape[0]),
                        message='log_prior_prob_mu class method assumes this fact is true')

    @property
    def name(self):
        return 'VampPriorMixture'

    def qmu(self, encoder: callable, **kwargs) -> tfpd.Distribution:
        return encoder(self.u, **kwargs)

    def log_prior_prob_mu(self, **kwargs) -> tf.Tensor:
        qmu = self.qmu(**kwargs)
        expected_ll = 0.5 * tf.linalg.logdet(self.prior_mu.covariance() / 2 / np.pi)
        expected_ll -= 0.5 * tf.reduce_sum(qmu.mean() ** 2, axis=1)
        expected_ll -= 0.5 * tf.linalg.trace(qmu.covariance())
        return expected_ll

    def pz_c(self, **kwargs) -> tfpd.Distribution:
        qmu = self.qmu(**kwargs)
        L = tf.linalg.cholesky(tf.linalg.cholesky_solve(self.L, tf.eye(self.latent_dim)) + qmu.covariance())
        return tfpd.MultivariateNormalTriL(qmu.mean(), L)

    def log_prob_z_c(self, z: tf.Tensor, **kwargs) -> tf.Tensor:
        qmu = self.qmu(**kwargs)
        P = self.L @ tf.transpose(self.L, [0, 2, 1])
        d = z - qmu.mean()
        expected_ll = 0.5 * tf.linalg.logdet(P / 2 / np.pi)
        expected_ll -= 0.5 * (tf.expand_dims(d, axis=-2) @ P @ tf.expand_dims(d, axis=-1))[:, :, 0, 0]
        expected_ll -= 0.5 * tf.linalg.trace(P @ qmu.covariance())
        return expected_ll


class VampPrior(Prior):
    def __init__(self, *, latent_dim: int, num_clusters: int, u: Union[U, Tuple[U]], **kwargs):
        super().__init__(latent_dim=latent_dim, num_clusters=num_clusters, **kwargs)
        self.u = u

    @property
    def name(self):
        return 'VampPrior'

    def pz_c(self, encoder: callable, **kwargs) -> tfpd.Distribution:
        return encoder(self.u, **kwargs)

    def pz(self, encoder: callable, **kwargs) -> tfpd.Distribution:
        pz_c = self.pz_c(encoder, **kwargs)
        return tfp.distributions.MixtureSameFamily(tfpd.Categorical(logits=tf.zeros(pz_c.batch_shape)), pz_c)

    def kl_divergence(self, qz_x: tfpd.Distribution, **kwargs):
        return -qz_x.entropy() - self.pz(**kwargs).log_prob(qz_x.sample())

    def cluster_probabilities(self, *, samples, **kwargs) -> tf.Tensor:
        return tf.nn.softmax(self.pz_c(**kwargs).log_prob(tf.expand_dims(samples, axis=1)))


def select_prior(prior: str, **kwargs):
    if prior == 'StandardNormal':
        return StandardNormal(**kwargs)
    elif prior == 'BetaStickBreaking':
        return BetaStickBreaking(**kwargs)
    elif prior == 'GaussianMixture':
        return GaussianMixture(**kwargs)
    elif prior == 'VampPriorMixture':
        return VampPriorMixture(**kwargs)
    elif prior == 'VampPrior':
        return VampPrior(**kwargs)
    else:
        return Prior(**kwargs)


def clean_prior_name(prior: str) -> str:
    if prior == 'StandardNormal':
        return '$\\mathcal{N}(0,I)$'
    elif prior == 'GaussianMixture':
        return 'GMM'
    elif prior == 'VampPriorMixture':
        return 'VMM'
    elif prior == 'VampPrior':
        return 'VampPrior'  #~\\citep{tomczak_vae_2018}'
    else:
        return prior
