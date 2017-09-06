from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

class ModelTrainer:

	def __init__(self):
		self.arch_exists = False
        pass

	def train(self, X, Y):
		if not self.arch_exists:
			self._create_architecture()

		self.model.fit(X, Y, batch_size=200, nb_epoch=10, verbose=1)

	def _create_architecture(self):
		model = Sequential()
		model.add(BatchNormalization(axis=1, input_shape=(11,)))
		model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(11,)))
		model.add(Convolution2D(6, 3, 3, activation='relu'))
		model.add(Dropout(.25))
		model.add(Flatten())
		model.add(Dense(3, activation=None))
		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

		self.model = model
