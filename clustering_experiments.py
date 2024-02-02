import argparse
import os
import itertools
import pickle
import json
import zlib

import pandas as pd
import tensorflow as tf

from callbacks import PerformanceMonitor
from datasets import load_tensorflow_dataset
from models import build_encoder, build_decoder, select_model
from priors import select_prior
from utils import sample_data_indices, clustering_performance

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--max_epochs', type=int, default=10000)
parser.add_argument('--mode', type=str)
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--replace', action='store_true', default=False, help='force saved model replacement')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

# check arguments
assert args.mode in {'tuning', 'testing'}

# enable GPU determinism
tf.config.experimental.enable_op_determinism()

# make experiment directory
exp_path = os.path.join('experiments', 'clustering', str(args.seed), args.dataset, 'dim_{:d}'.format(args.latent_dim))
os.makedirs(exp_path, exist_ok=True)

# load dataset
x_train, labels_train, x_valid, labels_valid = load_tensorflow_dataset(args.dataset)
num_classes = int(tf.shape(tf.unique(labels_train)[0])[0])

# tuning mode test cases
if args.mode == 'tuning':
    test_cases = [
        dict(model='EmpiricalBayesVariationalAutoencoder',
             model_kwargs=dict(batch_size=bs, learning_rate=args.learning_rate),
             prior='VampPriorMixture',
             prior_kwargs=dict(num_clusters=100, inference='MAP-DP', prior_learning_ratio=plr))
        for bs, plr in list(itertools.product([128, 256, 512, 1024], [1.0]))]

# testing mode test cases
elif args.mode == 'testing':
    try:
        df = pd.read_pickle(os.path.join(exp_path, 'tuning_performance.pkl'))
        df = df.groupby(level=['batch_size', 'learning_rate', 'prior_learning_ratio']).mean()
        bs, lr, plr = df['NMI (unmerged)'].idxmax()
        batch_size, learning_rate, prior_learning_ratio = int(bs), float(lr), float(plr)
    except FileNotFoundError:
        raise FileNotFoundError('please run script in tuning mode first')

    # models, priors, and configurations to test
    test_cases = [
        dict(model='VariationalAutoencoder',
             model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
             prior='VampPrior',
             prior_kwargs=dict(num_clusters=100, inference=None, prior_learning_ratio=None)),
        dict(model='VariationalDeepEmbedding',
             model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
             prior='GaussianMixture',
             prior_kwargs=dict(num_clusters=num_classes, inference=None, prior_learning_ratio=None)),
        # dict(model='EmpiricalBayesVariationalAutoencoder',
        #      model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
        #      prior='GaussianMixture',
        #      prior_kwargs=dict(num_clusters=num_classes, inference='MLE', prior_learning_ratio=prior_learning_ratio)),
        dict(model='EmpiricalBayesVariationalAutoencoder',
             model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
             prior='GaussianMixture',
             prior_kwargs=dict(num_clusters=100, inference='MAP', prior_learning_ratio=prior_learning_ratio)),
        dict(model='EmpiricalBayesVariationalAutoencoder',
             model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
             prior='GaussianMixture',
             prior_kwargs=dict(num_clusters=100, inference='MAP-DP', prior_learning_ratio=prior_learning_ratio)),
        # dict(model='EmpiricalBayesVariationalAutoencoder',
        #      model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
        #      prior='VampPriorMixture',
        #      prior_kwargs=dict(num_clusters=num_classes, inference='MLE', prior_learning_ratio=prior_learning_ratio)),
        dict(model='EmpiricalBayesVariationalAutoencoder',
             model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
             prior='VampPriorMixture',
             prior_kwargs=dict(num_clusters=100, inference='MAP', prior_learning_ratio=prior_learning_ratio)),
        dict(model='EmpiricalBayesVariationalAutoencoder',
             model_kwargs=dict(batch_size=batch_size, learning_rate=learning_rate),
             prior='VampPriorMixture',
             prior_kwargs=dict(num_clusters=100, inference='MAP-DP', prior_learning_ratio=prior_learning_ratio)),
    ]

else:
    raise NotImplementedError

# loop over the trials and test cases
performance = pd.DataFrame()
elbo_surgery = pd.DataFrame()
for trial in range(1, 1 + args.num_trials):
    for i, test_case in enumerate(test_cases):
        print('*** Trial {:d}/{:d} |  Case {:d}/{:d} ***'.format(trial, args.num_trials, i + 1, len(test_cases)))

        # a deterministic but seemingly random transformation of the experiment seed into a trial seed
        trial_seed = int(zlib.crc32(str(trial * (args.seed or 1)).encode())) % (2 ** 32 - 1)
        tf.keras.utils.set_random_seed(trial_seed)

        # VampPrior pseudo-input initialization
        u_init = tf.gather(x_train, sample_data_indices(x_train, test_case['prior_kwargs']['num_clusters']))

        # select prior
        prior_lr = test_case['prior_kwargs'].get('prior_learning_ratio')
        prior_lr = None if prior_lr is None else test_case['model_kwargs']['learning_rate'] * prior_lr
        latent_prior = select_prior(test_case['prior'], **test_case['prior_kwargs'], **dict(
            latent_dim=args.latent_dim,
            learning_rate=prior_lr,
            u=tf.Variable(u_init, name='u'),
        ))

        # construct and compile model
        model = select_model(test_case['model'])(
            encoder=build_encoder(dim_x=x_train.shape.as_list()[1:]),
            decoder=build_decoder(latent_dim=args.latent_dim, dim_x=x_train.shape.as_list()[1:]),
            prior=latent_prior,
            **test_case['model_kwargs'],
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(test_case['model_kwargs']['learning_rate']))

        # model save path
        # test_case = copy.deepcopy(test_case)
        # if hasattr(test_case['learning_rate'], 'get_config'):
        #     test_case['learning_rate'] = test_case['learning_rate'].get_config()
        save_path = os.path.join(exp_path, 'trial_{:d}'.format(trial), test_case['model'], test_case['prior'])
        save_path = os.path.join(save_path, 'config_{:d}'.format(zlib.crc32(json.dumps(test_case).encode('utf-8'))))

        # if we are set to resume and the model directory already contains a saved model, load it
        if not bool(args.replace) and os.path.exists(os.path.join(save_path, 'checkpoint')):
            print('Loading existing model.')
            checkpoint = tf.train.Checkpoint(model)
            checkpoint.restore(os.path.join(save_path, 'best_checkpoint')).expect_partial()

        # otherwise, fit and save the model
        else:
            hist = model.fit(
                x=dict(x=x_train, label=labels_train),
                validation_data=dict(x=x_valid, label=labels_valid),
                batch_size=test_case['model_kwargs']['batch_size'],
                epochs=args.max_epochs,
                verbose=False,
                callbacks=[PerformanceMonitor(monitor='val_nmi', patience=100)] + model.callbacks(x=x_train))
            model.save_weights(os.path.join(save_path, 'best_checkpoint'))
            with open(os.path.join(save_path, 'history.pkl'), 'wb') as f:
                pickle.dump(hist.history, f)

        # configuration index
        index = pd.MultiIndex.from_frame(pd.DataFrame({
            **{'model': [test_case['model']]},
            **test_case['model_kwargs'],
            **{'prior': [test_case['prior']]},
            **test_case['prior_kwargs'],
            **{'save_path': save_path}
        }))

        # save clustering performance
        tf.keras.utils.set_random_seed(trial_seed)
        all_data = dict(x=tf.concat([x_train, x_valid], axis=0), label=tf.concat([labels_train, labels_valid], axis=0))
        latent_samples = model.predict(all_data, verbose=False)
        cluster_probs = model.cluster_probabilities(latent_samples)
        performance = pd.concat([performance, clustering_performance(cluster_probs, all_data['label'], index)])
        performance.to_pickle(os.path.join(exp_path, args.mode + '_performance.pkl'))

        # save model performance
        if args.mode == 'testing':
            tf.keras.utils.set_random_seed(trial_seed)
            elbo_surgery = pd.concat([elbo_surgery, pd.DataFrame(model.elbo_surgery(dict(x=x_valid)), index=index)])
            elbo_surgery.to_pickle(os.path.join(exp_path, 'elbo_surgery.pkl'))

        # clear model from memory
        tf.keras.backend.clear_session()
