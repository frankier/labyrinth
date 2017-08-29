from collections import defaultdict
import pickle
import numpy as np
import argparse
import random
import logging
import sys
import math

import gym
from gym import wrappers

import labyrinth  # noqa
from labyrinth.board import get_possible_actions, get_board_reachability


def print_reachability(board, position):
    reachability = get_board_reachability(board, position)
    print(np.transpose(reachability))


class ReplAgent(object):
    """AI hack - try using a human!"""
    def __init__(self, wrapper):
        self.env = wrapper.env

    def act(self, observation, reward, done):
        from labyrinth.env import LabyrinthState
        while True:
            inp = input()
            if inp.startswith("ob"):
                print(observation)
            elif inp.startswith("re"):
                print(reward)
            elif inp.startswith("do"):
                print(done)
            elif inp.startswith("po"):
                possible_actions = observation.get_possible_actions()
                print(possible_actions)
            elif inp.startswith("ma"):
                print_reachability(observation.board_state[0],
                                   observation.current_position())
            elif inp.startswith("pu"):
                push = eval(inp.split(" ", 1)[1])
                new_board_state, new_players = observation.act_push(push)
                state = LabyrinthState(
                    new_board_state,
                    observation.player_turn,
                    new_players,
                    observation.num_treasures)
                print(state)
                print_reachability(state.board_state[0],
                                   state.current_position())
            else:
                action = eval(inp)
                break
        return action


class NaiveSampler(object):
    def __init__(self, env):
        self.action_space = env.action_space

    def sample(self, ob):
        return self.action_space.sample()


class PossibleSampler(object):
    def __init__(self, env):
        self.action_space = env.action_space

    def sample(self, ob):
        return random.choice(ob.get_possible_actions())


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env, sampler):
        self.sampler = sampler(env)

    def act(self, observation, reward, done):
        return self.sampler.sample(observation)


def argmax(d):
    max_k = None
    max_v = -math.inf
    for k, v in d.items():
        if v >= max_v:
            max_k = k
            max_v = v
    return max_k


class TabularQAgent(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, env, **userconfig):
        self.sampler = PossibleSampler(env)
        self.config = {
            "init_mean": 0.0,      # Initialize Q values with this mean
            "init_dev": 0.0,       # Initialize Q values with this individual deviation
            "learning_rate": 0.1,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.95}        # Number of iterations
        self.config.update(userconfig)
        self.q = {}

    def add_q_values(self, action_dict, ob):
        for action in ob.get_possible_actions():
            action_dict[action] = \
                random.gauss(self.config["init_mean"],
                             self.config["init_dev"])

    def ensure_action_dict(self, item, ob):
        if item not in self.q:
            action_dict = {}
            self.add_q_values(action_dict, ob)
            self.q[item] = action_dict

    def load_model(self, fn):
        self.q = pickle.load(open(fn, 'rb'))

    def save_model(self, fn):
        pickle.dump(self.q, open(fn, 'wb'))

    def act(self, observation, reward, done, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        if np.random.random() > eps:
            item = observation.item()
            self.ensure_action_dict(item, observation)
            action_dict = self.q[item]
            action = argmax(action_dict)
            if action:
                return action
        return self.sampler.sample(observation)

    def learn(self, env):
        config = self.config
        obs = env.reset()
        q = self.q

        reward = 0
        done = False
        while 1:
            action = self.act(obs, reward, done)
            obs2, reward, done, _ = env.step(action)
            future = 0.0
            item = obs2.item()
            self.ensure_action_dict(item, obs2)
            if not done:
                future = max(q[item].values())
            q[item][action] -= \
                self.config["learning_rate"] * \
                (q[item][action] - reward - config["discount"] * future)

            obs = obs2

            if done:
                return


agents = {
    'randumb': lambda env: RandomAgent(env, NaiveSampler),
    'random': lambda env: RandomAgent(env, PossibleSampler),
    'repl': ReplAgent,
    'qtab': TabularQAgent,
}


def run(agent, env, episode_count):
    for i in range(episode_count):
        reward = 0
        done = False
        print()
        print(" ** Episode", i, " ** ")
        ob = env.reset()
        while True:
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break


def run_training(agent, env, episode_count,
                 save_every=None, save_prefix="saved."):
    for t in range(episode_count):
        if save_every and t > 0 and t % save_every == 0:
            print("Saving")
            agent.save_model(save_prefix + str(t))
        print()
        print(" ** Episode", t, " ** ")
        agent.learn(env)


def main(env_id, agent, episode_count, outdir,
         seed=None, load=None, save=None, learn=False,
         save_every=None, save_prefix=None):
    if learn and not (save_every and save_prefix):
        assert False, \
            "If learn is true, save_every and save_prefix are manditory"
    env = gym.make(env_id)

    env = wrappers.Monitor(env, directory=outdir)
    if seed:
        env.seed(seed)
    else:
        env.seed()
    agent = agents[agent](env)
    if load:
        agent.load_model(load)

    if learn:
        run_training(agent, env, episode_count, save_every, save_prefix)
    else:
        run(agent, env, episode_count)

    # Close the env and write monitor result info to disk
    env.close()
    if save:
        agent.save_model(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('agent', help='Select the agent to run')
    parser.add_argument('env_id', nargs='?', default='Labyrinth3x3-tr0-v0',
                        help='Select the environment to run')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--episode-count', type=int, default=100,
                        help='Load agent from this file before running')
    parser.add_argument('--load',
                        help='Load agent model from this file before running')
    parser.add_argument('--save',
                        help='Save agent model to this file after running')
    parser.add_argument('--save-prefix', default="saved.",
                        help='Save agent model to this prefix every --save-every')
    parser.add_argument('--save-every', type=int,
                        help='Save agent model every x iterations')
    parser.add_argument('--learn', action="store_true",
                        help='Do agent model learning')
    parser.add_argument('--outdir', help='Where to save statistics and recordings')
    parser.add_argument('--quiet',
                        help='Only print out slow progress (every 100 iters)')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = '/tmp/gym-{}-{}-{}'.format(args.env_id, args.agent, random.randint(0, 100000))

    main(args.env_id, args.agent, args.episode_count, outdir,
         seed=args.seed, load=args.load, save=args.save, learn=args.learn,
         save_every=args.save_every, save_prefix=args.save_prefix)
