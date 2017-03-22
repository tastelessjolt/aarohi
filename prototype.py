import wave
import numpy as np
import tensorflow as tf

# TIMESTEPS =
# RNN_LAYERS =
# DENSE_LAYERS =
# TRAINING_STEPS =
# BATCH_SIZE =

class generator:

	w = 5;

	def read_int_array(self,filename):
		self.f = wave.open(filename, "rb");
		self.data = self.f.readframes(self.f.getnframes())

	def construct_data_set(self):
		x = []
		y = []

		for i in range(self.f.getnframes()-6):
			x.append(self.data[i:i+5])
			y.append(self.data[i+6])

		self.X = x
		self.Y = y

	def input_fn_train(self) :
		x = tf.Variable(self.X)
		y = tf.Variable(self.Y)
		return [x, y]

	def train(self):

		sparse_feature = tf.contrib.layers.sparse_column_with_integerized_feature("1", bucket_size=255)
		sparse_feature2 = tf.contrib.layers.sparse_column_with_integerized_feature("2", bucket_size=255)
		# sparse_feature3 = tf.contrib.layers.sparse_column_with_integerized_feature("3", bucket_size=255)
		# sparse_feature4 = tf.contrib.layers.sparse_column_with_integerized_feature("4", bucket_size=255)
		# sparse_feature5 = tf.contrib.layers.sparse_column_with_integerized_feature("5", bucket_size=255)

		estimator = tf.contrib.learn.DNNRegressor(
			feature_columns=[sparse_feature],
			hidden_units=[1024, 512, 256],
			optimizer=tf.train.ProximalAdagradOptimizer(
		    learning_rate=0.1,
		    l1_regularization_strength=0.001
		))

		estimator.fit(input_fn = self.input_fn_train)


		# regressor = tf.contrib.learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
		# 	n_classes=0,
		# 	verbose=1,
		# 	steps=TRAINING_STEPS,
		# 	optimizer='Adagrad',
		# 	learning_rate=0.03,
		# 	batch_size=BATCH_SIZE)

		regressor.fit(self.X, self.Y)

	def synthesize():

		seed = zeros()

		a = 1

a = generator()
a.read_int_array("mus.wav")
a.construct_data_set()
a.train()
a.synthesize()
