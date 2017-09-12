from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers.recurrent import LSTM

import numpy as np
import pickle

class ModelTrainer:

    def __init__(self):
	self.arch_exists = False
        pass

    def _preprocess(self, X):
        cols = X.shape[1]
        feature_means = []
        feature_stds = []
        for i in range(cols):
            col_mean = np.mean(X[:,i])
            col_std = np.std(X[:,i])
            feature_means.append(col_mean)
            feature_stds.append(col_std)
            X[:,i] = X[:,i] - col_mean
            X[:,i] = X[:,i] / (col_std + 1e-6)

        print(len(feature_means))
        
        norm_info = {'means': feature_means, 'stds': feature_stds}
        f = open("norm_info.sav", "w")
        pickle.dump(norm_info, f)
        f.close()
        return X
    
    def train(self, X, Y):
	if not self.arch_exists:
	    self._create_architecture()

        X = self._preprocess(X)

        filepath="model.{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, verbose=1, mode='auto')
        callbacks = [checkpoint]
        training_history = self.model.fit(X, Y, callbacks=callbacks, batch_size=100, epochs=200, verbose=1)

    def save(self, path):
        self.model.save(path)

    def _create_architecture(self):
	    model = Sequential()
            model.add(Dense(64, activation='tanh', batch_input_shape=(None,111)))
            model.add(Dense(64, activation='tanh'))
            model.add(Dense(8))
            model.compile(loss='mean_squared_error', optimizer=Adam(lr=.001, epsilon=1e-8), metrics=['accuracy'])
            self.model = model
