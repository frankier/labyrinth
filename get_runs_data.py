import argparse
import pickle
from os.path import join as pjoin

import gym
import pandas as pd

import agents

BOOTSTRAP_ITERS = 1000
EPISODES = 100


def main(modelbase, basedir, times, step, start=0):
    quartiles = ([], [], [])

    for test_run in range(times):
        train_iters = test_run * step + start
        outdir = basedir + '.' + str(train_iters)
        # Run the thing
        agents.main(
            'Labyrinth3x3-tr0-v0', 'qtab', EPISODES, outdir,
            load=modelbase + str(train_iters), quiet=True)

        # Load data from outdir
        results = gym.monitoring.load_results(outdir)
        episode_lengths = pd.Series(results['episode_lengths'])

        # Do some bootstrap resampling
        quartiles_inner = ([], [], [])
        for bootstrap_iter in range(BOOTSTRAP_ITERS):
            samp = episode_lengths.sample(50, replace=True)
            for quart in range(3):
                quantile = (quart + 1) * 0.25
                quartiles_inner[quart].append(samp.quantile(quantile))
        for quart in range(3):
            quartiles_inner[quart].sort()
            mid = (quartiles_inner[quart][499] +
                   quartiles_inner[quart][500]) / 2
            pos_err = quartiles_inner[quart][-25] - mid
            neg_err = mid - quartiles_inner[quart][24]
            quartiles[quart].append((mid, pos_err, neg_err))

    pickle.dump(quartiles, open(pjoin(basedir, 'quartiles.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('modelbase', help='Where to load the models from')
    parser.add_argument('basedir', help='Where to save the test run data to')
    parser.add_argument('times', type=int, help='How many models to load')
    parser.add_argument('step', type=int,
                        help='How many iterations between models')
    parser.add_argument('start', type=int, default=0, nargs="?",
                        help='How many iterations to star from')
    args = parser.parse_args()

    main(args.modelbase, args.basedir, args.times, args.step, args.start)
