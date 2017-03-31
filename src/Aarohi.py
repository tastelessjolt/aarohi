import tensorflow as tf
import numpy as np
import wave
import os

CELLS_IN_LSTM = 100
CASCADED_CELLS = 4
BATCH_SIZE = 5
MEL_BANDS = 5
SAMPLING_RATE = 44100
INPUT_SEC = 3
FEATURES_PER_BAND = SAMPLING_RATE * 4 * INPUT_SEC
MAX_SEQ_LENGTH = SAMPLING_RATE * 4 * 360
N_CLASSES = 256
OUTPUT_SEC = 0.5
OUTPUT_SEQ_LENGTH = SAMPLING_RATE * 4 * OUTPUT_SEC

class Aarohi():
	"""docstring for Aarohi"""
	def __init__(self):

		cell = tf.contrib.rnn.LSTMCell(CELLS_IN_LSTM, state_is_tuple=True)
		network = tf.contrib.rnn.MultiRNNCell([cell] * CASCADED_CELLS)
		
		data  = tf.placeholder(tf.float32, [None, FEATURES_PER_BAND, MEL_BANDS])
		target = tf.placeholder(tf.float32, [None, N_CLASSES, OUTPUT_SEQ_LENGTH, MEL_BANDS])

		output, state = tf.nn.dynamic_rnn(network, data, dtype = tf.float32)
		
		max_len = int(target.get_shape()[1])
		out_size = target.get_shape()[2:]

		out_dim = [CELLS_IN_LSTM]
		out_dim.extend(out_size)
		weight = tf.Variable(tf.truncated_normal([CELLS_IN_LSTM, out_size], stddev=0.1))
		bias = tf.Variable(tf.constant(0.1, shape = out_size))

		output = tf.reshape(output, [-1, CELLS_IN_LSTM])
		prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
		pred_dim = [-1, max_len]
		pred_dim.extend(out_size)
		prediction = tf.reshape(prediction, pred_dim)

		cross_entropy = -tf.reduce_sum(target*tf.log(prediction), [1,2])
		cross_entropy = tf.reduce_mean(cross_entropy)

		optimiser = tf.train.AdamOptimiser()
		minimize = optimiser.minimize(cross_entropy)

		mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

	def setTrainingData(self, filespath):

		train_data = []
		for path, dirs, files in os.walk(filespath):
			for file in files:
				print(file)
				if ( file[len(file)-3:] == "wav" ):
					f = wave.open(file, 'rb')
					dat = f.readframes(f.getnframes())

				f.close()
				train_data.append(dat)

		self.train_data_raw = train_data

	def train(self):

		init_op = tf.initialize_all_variables()
		sess = tf.Session()
		# sess.run(init_op)

		# no_of_batches = int(len(self.train_input)/BATCH_SIZE)
		# epoch = 5000
		# for i in range(epoch):
		    # ptr = 0
		    # for j in range(no_of_batches):
		        # inp, out = self.train_input[ptr:ptr+BATCH_SIZE], self.train_output[ptr:ptr+BATCH_SIZE]
		        # ptr+=BATCH_SIZE
		        # sess.run(minimize,{data: inp, target: out})
		    # print ("Epoch - ",str(i))

	def inventSong(self):
		pass