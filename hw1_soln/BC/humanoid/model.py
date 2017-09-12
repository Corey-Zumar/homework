from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant

import pickle
import numpy as np

class ModelTrainer:

	def __init__(self):
		self.arch_exists = False
        pass

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
		if not self.arch_exists:
			self._create_architecture()
                X = self.preprocess(X)
                filepath="model.{epoch:02d}.hdf5"
                checkpoint = ModelCheckpoint(filepath, verbose=1, mode='auto')
		callbacks = [checkpoint]
                training_history = self.model.fit(X, Y, callbacks=callbacks, batch_size=20, nb_epoch=200, verbose=1)
                return training_history

        def save(self, path):
            self.model.save(path)

        def _create_architecture(self):
		model = Sequential()
                model.add(Dense(64, activation='tanh', batch_input_shape=(None,376)))
                model.add(Dense(64, activation='tanh'))
                model.add(Dense(17, activation=None))
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                self.model = model

