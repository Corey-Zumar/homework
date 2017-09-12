import gym
import keras
import pickle
import numpy as np

class Model:
	def __init__(self, model_path, mean_info_path):
		print(model_path)
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

	def save(self, path):
		self.model.save(path)

	def preprocess(self, X):
		cols = X.shape[1]
		feature_means = []
		feature_stds = []
		for i in range(0, cols):
			col_mean = np.mean(X[:,i])
			col_std = np.std(X[:,i])
			feature_means.append(col_mean)
			feature_stds.append(col_std)
			X[:,i] = X[:,i] - col_mean
			X[:,i] = X[:,i] / (col_std + 1e-6)
	        
			norm_info = {'means': feature_means, 'stds': feature_stds}
			f = open("norm_info.sav", "w")
			pickle.dump(norm_info, f)
			f.close()
			return X
        
	def train(self, X, Y):
		X = self.preprocess(X)
		self.model.fit(X, Y)

def main():
	import argparse
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--num_rollouts', type=int, default=20,
							help='Number of expert roll outs')
	parser.add_argument('--model_path', type=str)
	parser.add_argument('--mean_path', type=str)
	parser.add_argument('--train_path', type=str)
	parser.add_argument('--model_data_path', type=str)
	args = parser.parse_args()

	model = Model(args.model_path, args.mean_path)

	if args.train_path:
		train_file = open(args.train_path, "rb")
		train_data = pickle.load(train_file)
		train_file.close()
		train_obs = train_data['observations']
		train_actions = train_data['actions']

		model.train(train_obs, train_actions)
		model.save(args.model_path)

	env = gym.make('Walker2d-v1')
	max_steps = env.spec.timestep_limit

	returns = []
	observations = []
	model_actions = []
	for i in range(args.num_rollouts):
		obs = env.reset()
		totalr = 0.
		steps = 0
		done = False
		while not done:
			model_action = model.predict(obs[None,:])
			observations.append(obs)
			model_actions.append(model_action)
			obs, r, done, _ = env.step(model_action)
			totalr += r
			steps += 1

			if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
			if steps >= max_steps:
				break
			returns.append(totalr)

	observations = np.array(observations)
	model_actions = np.array(model_actions)

	print('returns', returns)
	print('mean return', np.mean(returns))
	print('std of return', np.std(returns))

	model_data = {'observations': np.array(observations),
                   'actions': np.array(model_actions)}

	out_file = open(args.model_data_path, "w")
	pickle.dump(model_data, out_file)
	out_file.close()

if __name__ == "__main__":
	main()