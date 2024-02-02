import argparse
import json
import os
import pickle
import zlib

import pandas as pd
import tensorflow as tf

from callbacks import PerformanceMonitor
from datasets import load_tensorflow_dataset
from models import build_encoder, build_decoder, select_model
from priors import select_prior
from utils import sample_data_indices

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--num_trials', type=int, default=5)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--prior_learning_ratio', type=float, default=1.0)
parser.add_argument('--replace', action='store_true', default=False, help='force saved model replacement')
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

# enable GPU determinism
tf.config.experimental.enable_op_determinism()

# make experiment directory
exp_path = os.path.join('experiments', 'elbo-surgery', str(args.seed), args.dataset)
os.makedirs(exp_path, exist_ok=True)

# load dataset
x_train, labels_train, x_valid, labels_valid = load_tensorflow_dataset(args.dataset)

# elbo surgery test cases
test_cases = [
    dict(model='VariationalAutoencoder',
         prior='StandardNormal',
         prior_kwargs=dict(num_clusters=1, inference=None)),
    dict(model='VariationalAutoencoder',
         prior='VampPrior',
         prior_kwargs=dict(num_clusters=100, inference=None)),
    dict(model='EmpiricalBayesVariationalAutoencoder',
         prior='GaussianMixture',
         prior_kwargs=dict(num_clusters=100, inference='MAP-DP')),
    dict(model='EmpiricalBayesVariationalAutoencoder',
         prior='VampPriorMixture',
         prior_kwargs=dict(num_clusters=100, inference='MAP-DP')),]

# loop over the trials and test cases
elbo_surgery = pd.DataFrame()
for trial in range(1, args.num_trials + 1):
    trial_path = os.path.join(exp_path, 'trial_{:d}'.format(trial))
    for i, test_case in enumerate(test_cases):
        print('*** Trial {:d}/{:d} | Model {:d}/{:d} ***'.format(trial, args.num_trials, i + 1, len(test_cases)))

        # a deterministic but seemingly random transformation of the experiment seed into a trial seed
        trial_seed = int(zlib.crc32(str(trial * (args.seed or 1)).encode())) % (2 ** 32 - 1)
        tf.keras.utils.set_random_seed(trial_seed)

        # VampPrior pseudo-input initialization
        u_init = tf.gather(x_train, sample_data_indices(x_train, test_case['prior_kwargs']['num_clusters']))

        # select prior
        latent_prior = select_prior(test_case['prior'], **test_case['prior_kwargs'], **dict(
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate * args.prior_learning_ratio,
            u=tf.Variable(u_init, name='u'),
        ))

        # construct and compile model
        model = select_model(test_case['model'])(
            encoder=build_encoder(dim_x=x_train.shape.as_list()[1:]),
            decoder=build_decoder(latent_dim=args.latent_dim, dim_x=x_train.shape.as_list()[1:]),
            prior=latent_prior,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

        # model save path
        save_path = os.path.join(trial_path, test_case['model'], test_case['prior'])
        save_path = os.path.join(save_path, 'config_{:d}'.format(zlib.crc32(json.dumps(test_case).encode('utf-8'))))

        # if we are set to resume and the model directory already contains a saved model, load it
        if not bool(args.replace) and os.path.exists(os.path.join(save_path, 'checkpoint')):
            print('Loading existing model.')
            checkpoint = tf.train.Checkpoint(model)
            checkpoint.restore(os.path.join(save_path, 'best_checkpoint')).expect_partial()
            with open(os.path.join(save_path, 'history.pkl'), 'rb') as f:
                hist = pickle.load(f)

        # otherwise, fit and save the model
        else:
            hist = model.fit(
                x=dict(x=x_train, label=labels_train),
                validation_data=dict(x=x_valid, label=labels_valid),
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                verbose=False,
                callbacks=[PerformanceMonitor(monitor='val_elbo', patience=args.patience)])
            model.save_weights(os.path.join(save_path, 'best_checkpoint'))
            with open(os.path.join(save_path, 'history.pkl'), 'wb') as f:
                pickle.dump(hist.history, f)
            hist = hist.history

        # perform ELBO surgery
        df = pd.DataFrame(
            data={**model.elbo_surgery(dict(x=x_valid)), **{'Epochs': len(hist['val_elbo']) - args.patience}},
            index=pd.MultiIndex.from_frame(pd.DataFrame({
                **{'model': [test_case['model']]},
                **{'prior': [test_case['prior']]},
                **{'latent_dim': [args.latent_dim]},
                **test_case['prior_kwargs'],
            })))
        elbo_surgery = pd.concat([elbo_surgery, df])
        elbo_surgery.to_pickle(os.path.join(exp_path, 'elbo_surgery.pkl'))

        # clear model from memory
        tf.keras.backend.clear_session()
