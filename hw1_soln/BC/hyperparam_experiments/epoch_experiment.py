import pickle
import tensorflow as tf
import numpy as np
import gym
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers.recurrent import LSTM

class Model:

	def __init__(self):
		self.arch_exists = False
		pass

	def _preprocess(self, X):
		X = np.copy(X)
		cols = X.shape[1]
		self.feature_means = []
		self.feature_stds = []
		for i in range(cols):
			col_mean = np.mean(X[:,i])
			col_std = np.std(X[:,i])
			self.feature_means.append(col_mean)
			self.feature_stds.append(col_std)
			X[:,i] = X[:,i] - col_mean
			X[:,i] = X[:,i] / (col_std + 1e-6)

		return X
	
	def train(self, X, Y):
		if not self.arch_exists:
		   self._create_architecture()

		X = self._preprocess(X)
		self.model.fit(X, Y, batch_size=200, epochs=1, verbose=1)

	def save(self, path):
		self.model.save(path)

	def predict(self, data):
		data = np.copy(data)
		means = self.feature_means
		stds = self.feature_stds
		for i in range(0, data.shape[1]):
			data[:,i] = data[:,i] - means[i]
			data[:,i] = data[:,i] / (stds[i] + 1e-6)

		return self.model.predict(data, batch_size=1, verbose=1)

	def _create_architecture(self):
		model = Sequential()
		model.add(Dense(64, activation='tanh', batch_input_shape=(None,111)))
		model.add(Dense(64, activation='tanh'))
		model.add(Dense(8))
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=.001, epsilon=1e-8), metrics=['accuracy'])
		self.model = model
		self.arch_exists = True

def eval(model, num_rollouts=3):
	env = gym.make('Ant-v1')
	max_steps = env.spec.timestep_limit

	returns = []
	observations = []
	actions = []
	for i in range(num_rollouts):
		print('iter', i)
		obs = env.reset()
		done = False
		totalr = 0.
		steps = 0
		while not done:
			action = model.predict(obs[None,:])
			observations.append(obs)
			actions.append(action)
			obs, r, done, _ = env.step(action)
			totalr += r
			steps += 1
			if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
			if steps >= max_steps:
				break
		returns.append(totalr)

	returns_data = {'returns': returns, 'mean': np.mean(returns), 'std': np.std(returns)}

	print(returns_data)

	return returns_data

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str)
	parser.add_argument('--num_epochs', type=int)
	args = parser.parse_args()

	data_file = open(args.data_path, "rb")
	data = pickle.load(data_file)
	data_file.close()
	obs = data['observations']
	actions = np.array([item[0] for item in data['actions']])

	model = Model()

	all_returns = []

	for i in range(args.num_epochs):
		model.train(obs, actions)
		returns = eval(model)
		all_returns.append(returns)

	out_file = open("epoch_experiment_returns.sav", "w")
	pickle.dump(all_returns, out_file)
	out_file.close()
