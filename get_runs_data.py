import argparse
import pickle
from os.path import join as pjoin

import gym
import math

import agents

BOOTSTRAP_ITERS = 1000
EPISODES = 100


def gen_iters(step, max, base=2):
    """
    Takes step and max and generates log scale (lin between 0 and 1)
    Always includes 0 and max
    [(position, iters)]
    """
    yield (0, 0)
    for test_run in range(0, math.floor(math.log(max / step, base)) + 1):
        train_iters = step * base ** test_run
        yield (test_run + 1, train_iters)
    yield (math.log(max / step, base) + 1, max)


def main(modelbase, basedir, step, max):
    ys = []
    positions = []
    x = []

    for (position, train_iters) in gen_iters(step, max):
        positions.append(position)
        x.append(train_iters)

        outdir = basedir + '.' + str(train_iters)
        # Run the thing
        agents.main(
            'Labyrinth3x3-tr0-v0', 'qtab', EPISODES, outdir,
            load=modelbase + str(train_iters), quiet=True, eps=0.0)

        # Load data from outdir
        results = gym.monitoring.load_results(outdir)
        episode_lengths = results['episode_lengths']

        ys.append(episode_lengths)

    pickle.dump((x, ys, positions), open(pjoin(basedir, 'runs.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('modelbase', help='Where to load the models from')
    parser.add_argument('basedir', help='Where to save the test run data to')
    parser.add_argument('step', type=int,
                        help='How many iterations between models')
    parser.add_argument('max', type=int,
                        help='How many iteration the max should have')
    args = parser.parse_args()

    main(args.modelbase, args.basedir, args.step, args.max)
