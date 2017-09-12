import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant

import pickle
import numpy as np

class Model:
	def __init__(self):
		self.create_architecture()

	def create_architecture(self):
		model = Sequential()
		model.add(Dense(10, activation='tanh', batch_input_shape=(None,17)))
		model.add(Dense(10, activation='tanh'))
		model.add(Dense(6, activation=None))
		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
		self.model = model

	def predict(self, data):
	# 	means = self.feature_means
	# 	stds = self.feature_stds
	# 	for i in range(0, data.shape[1]):
	# 		data[:,i] = data[:,i] - means[i]
	# 		data[:,i] = data[:,i] / (stds[i] + 1e-6)

		return self.model.predict(data, batch_size=1, verbose=1)

	def preprocess(self, X):
		print(X.shape)
		cols = X.shape[1]
		self.feature_means = []
		self.feature_stds = []
		for i in range(0, cols):
			col_mean = np.mean(X[:,i])
			col_std = np.std(X[:,i])
			self.feature_means.append(col_mean)
			self.feature_stds.append(col_std)
			X[:,i] = X[:,i] - col_mean
			X[:,i] = X[:,i] / (col_std + 1e-6)

		return X
        
	def train(self, X, Y):
		#X = self.preprocess(X)
		self.model.fit(X, Y, batch_size=200, epochs=100, verbose=1)

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_policy_file', type=str)
	
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--num_rollouts', type=int, default=20,
							help='Number of expert roll outs')
	parser.add_argument('--data_path', type=str)
	parser.add_argument('--num_dagger_iters', type=str)
	args = parser.parse_args()

	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	data_file = open(args.data_path, "rb")
	data = pickle.load(data_file)
	data_file.close()

	with tf.Session():
		tf_util.initialize()
		import gym
		env = gym.make('Walker2d-v1')
		max_steps = env.spec.timestep_limit

		model = Model()

		for i in range(int(args.num_dagger_iters)):
			model.train(data['observations'], data['actions'])

			returns = []
			observations = []
			expert_actions = []
			for i in range(args.num_rollouts):
				obs = env.reset()
				totalr = 0.
				steps = 0
				done = False
				while not done:
					model_action = model.predict(obs[None,:])
					expert_action = policy_fn(obs[None,:])
					observations.append(obs)
					expert_actions.append(expert_action[0])
					obs, r, done, _ = env.step(model_action)
					totalr += r
					steps += 1

					if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
					if steps >= max_steps:
						break
					returns.append(totalr)

			observations = np.array(observations)
			expert_actions = np.array(expert_actions)

			data['actions'] = np.concatenate([data['actions'], expert_actions], axis=0)
			data['observations'] = np.concatenate([data['observations'], observations], axis=0)

			print('returns', returns)
			print('mean return', np.mean(returns))
			print('std of return', np.std(returns))

def run_standard():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_policy_file', type=str)
	
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--num_rollouts', type=int, default=20,
							help='Number of expert roll outs')
	parser.add_argument('--data_path', type=str)
	parser.add_argument('--num_dagger_iters', type=str)
	args = parser.parse_args()

	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	data_file = open(args.data_path, "rb")
	data = pickle.load(data_file)
	data_file.close()
	model = Model()

	with tf.Session():
		tf_util.initialize()
		import gym
		env = gym.make('Walker2d-v1')
		max_steps = env.spec.timestep_limit

		for i in range(int(args.num_dagger_iters)):
			model.train(data['observations'], data['actions'])
			returns = []
			observations = []
			expert_actions = []
			for i in range(args.num_rollouts):
				obs = env.reset()
				totalr = 0.
				steps = 0
				done = False
				while not done:
					model_action = model.predict(obs[None,:])
					expert_action = policy_fn(obs[None,:])
					observations.append(obs)
					expert_actions.append(expert_action[0])
					obs, r, done, _ = env.step(model_action)
					totalr += r
					steps += 1

					if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
					if steps >= max_steps:
						break
					returns.append(totalr)

			observations = np.array(observations)
			expert_actions = np.array(expert_actions)

			# data['actions'] = np.concatenate([data['actions'], expert_actions], axis=0)
			# data['observations'] = np.concatenate([data['observations'], observations], axis=0)

			print('returns', returns)
			print('mean return', np.mean(returns))
			print('std of return', np.std(returns))

if __name__ == "__main__":
	main()
	#run_standard()