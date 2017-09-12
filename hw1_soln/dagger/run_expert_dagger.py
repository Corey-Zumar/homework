import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('--model_data_path', type=str)
	parser.add_argument('--train_path', type=str)

	args = parser.parse_args()

	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	model_data_file = open(args.model_data_path, "rb")
	model_data = pickle.load(model_data_file)
	model_data_file.close()
	model_observations = model_data['observations']

	with tf.Session():
		tf_util.initialize()

		expert_actions = policy_fn(model_observations)

		train_data_file = open(args.train_path, "rb")
		train_data = pickle.load(train_data_file)
		train_data_file.close()

		obs = train_data['observations']
		new_obs = np.concatenate((obs, model_observations), axis=0)
		actions = train_data['actions']
		new_actions = np.concatenate((actions, expert_actions), axis=0)
		train_data['observations'] = new_obs
		train_data['actions'] = new_actions

		train_data_file = open(args.train_path, "w")
		pickle.dump(train_data, train_data_file)
		train_data_file.close()

if __name__ == "__main__":
	main()