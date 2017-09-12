import sys
import os
import argparse
import pickle
import numpy as np

from matplotlib import pyplot as plt

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--returns_path', type=str)
	args = parser.parse_args()

	returns_file = open(args.returns_path, "rb")
	returns_data = pickle.load(returns_file)
	returns_file.close()

	epochs = range(len(returns_data))
	returns_means = [item['mean'] for item in returns_data]

	plt.xticks(np.arange(0, len(epochs), 1.0))
	plt.title("Mean return in the Ants environment for a neural network \n as a function of the number of training epochs")
	plt.xlabel("Number of training epochs")
	plt.ylabel("Mean return for 3 rollouts")
	plt.plot(epochs, returns_means)
	plt.show()