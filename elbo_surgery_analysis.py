import argparse
import os

import pandas as pd

from models import clean_model_name
from priors import clean_prior_name
from utils import process_results


def clean_table(df):
    index_order = df.index.names
    df.reset_index(['Model'], inplace=True)
    df['Model'] = df['Model'].apply(clean_model_name)
    df.set_index('Model', append=True, inplace=True)
    df.reset_index(['Prior'], inplace=True)
    df['Prior'] = df['Prior'].apply(clean_prior_name)
    df.set_index('Prior', append=True, inplace=True)
    return df.reorder_levels(index_order)


def print_elbo_surgery_table(exp_path, threshold):
    # loop over datasets
    for dataset in os.listdir(exp_path):
        try:
            elbo_surgery = pd.read_pickle(os.path.join(exp_path, dataset, 'elbo_surgery.pkl'))
        except FileNotFoundError:
            continue
        if len(elbo_surgery) == 0:
            continue

        # reconfigure table
        assert elbo_surgery.index.get_level_values('latent_dim').nunique() == 1, 'assumptions violated'
        elbo_surgery = elbo_surgery.reset_index(['model', 'prior'], drop=False).reset_index(drop=True)
        elbo_surgery.rename(columns={'model': 'Model', 'prior': 'Prior'}, inplace=True)
        elbo_surgery.set_index(['Model', 'Prior'], inplace=True)

        # process results
        elbo_surgery['Negative Distortion'] = -elbo_surgery['Distortion']
        max_table = elbo_surgery[['log p(x|x)', 'log p(x)', 'Negative Distortion']].copy()
        max_table = process_results(max_table, threshold, mode='max', pm_std=True)
        no_test_cols = ['I[z;n]']
        min_table = elbo_surgery[['Avg. KL', 'Marginal KL'] + no_test_cols].copy()
        min_table = process_results(min_table, threshold, mode='min', pm_std=True, no_test_cols=no_test_cols)
        elbo_surgery = max_table.join(min_table)

        # print integration table
        elbo_surgery.rename(columns={
            'log p(x)': '$\\log p(x)$',
            'log p(x|x)': '$\\log p(x|x)$',
            'Negative Distortion': '$\\E_q[\\log p(x|z)]$',
            'Avg. KL': '$\\dkl(q(z|x)||p(z))$',
            'Marginal KL': '$\\dkl(q(z)||p(z))$',
            'I[z;n]': '$\\mathbb{I}[z;n]$',
        }, inplace=True)
        s = clean_table(elbo_surgery).style
        s.to_latex(os.path.join('results', 'elbo-surgery-{:s}.tex'.format(dataset)),
                   column_format='l' * elbo_surgery.index.nlevels + '|' + 'c' * len(elbo_surgery.columns),
                   hrules=True,
                   multirow_align='t')


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()

    # make sure results directory exists
    os.makedirs('results', exist_ok=True)

    # set experiment path and check if it exists
    experiment_path = os.path.join('experiments', 'elbo-surgery', str(args.seed))
    if not os.path.exists(experiment_path):
        exit(code='Experiment directory not found!')

    # elbo surgery table
    print_elbo_surgery_table(experiment_path, args.threshold)
