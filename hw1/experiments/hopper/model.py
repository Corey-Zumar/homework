from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM

class ModelTrainer:

	def __init__(self):
		self.arch_exists = False
        pass

	def train(self, X, Y):
		if not self.arch_exists:
			self._create_architecture()
		training_history = self.model.fit(X, Y, batch_size=200, nb_epoch=200, verbose=1)
		return training_history

        def save(self, path):
            self.model.save(path)

        def _create_architecture(self):
		model = Sequential()
		model.add(BatchNormalization(input_shape=(11,1)))
                #model.add(Dense(11, input_shape=(11,1)))	
		#model.add(Dense(8, activation='relu'))
                #model.add(Dense(6, activation='relu'))
                model.add(Convolution1D(8, 6, activation='relu'))
		model.add(Convolution1D(6, 2, activation='relu'))
	        model.add(Convolution1D(2, 2, activation='relu'))
		model.add(Flatten())
                model.add(Dense(3))
		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                self.model = model

