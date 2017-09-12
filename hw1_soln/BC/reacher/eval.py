#!/usr/bin/env python

"""
Code to perform behavioral cloning via a pre-trained policy

Modified from the included `run_expert` script by Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import gym
import keras

class Predictor:

    def __init__(self, model_path, mean_info_path):
        f = open(mean_info_path, "rb")
        self.mean_info = pickle.load(f)
        f.close()
        self.model = keras.models.load_model(model_path)

    def predict(self, data):
        means = self.mean_info['means']
        stds = self.mean_info['stds']
        for i in range(0, data.shape[1]):
           data[:,i] = data[:,i] - means[i]
           data[:,i] = data[:,i] / (stds[i] + 1e-6)

        return self.model.predict(data, batch_size=1, verbose=1)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--mean_path', type=str)
    args = parser.parse_args()

    print('loading and building expert policy')
    predictor = Predictor(args.model_path, args.mean_path)
    policy_fn = predictor.predict
    print('loaded and built')

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    return expert_data

if __name__ == '__main__':
    expert_data = main()
