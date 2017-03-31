import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wv
import os


def bf16(num):
	inum = bin(np.uint16(num))
	l = [int(b) for b in inum[2:]]
	p = [0]*(16-len(l))
	p.extend(l)
	return p

def f16b(bin_array):
	stri = "".join([str(x) for x in bin_array])
	if(stri[0] == '1'):
		return int(stri,2) - (1<<16)
	else:
		return int(stri,2)


CELLS_IN_LSTM = 100
CASCADED_CELLS = 4
BATCH_SIZE = 10000
MEL_BANDS = 1
SAMPLING_RATE = 96000
INPUT_SEC = 3
FEATURES_PER_BAND = SAMPLING_RATE * INPUT_SEC
MAX_SEQ_LENGTH = SAMPLING_RATE * 360
N_CLASSES = 16
OUTPUT_SEC = 0.5
OUTPUT_SEQ_LENGTH = SAMPLING_RATE * OUTPUT_SEC
EPOCH = 5000

class Aarohi():
	"""docstring for Aarohi"""
	def __init__(self):

		cell = tf.contrib.rnn.LSTMCell(CELLS_IN_LSTM, state_is_tuple=True)
		network = tf.contrib.rnn.MultiRNNCell([cell] * CASCADED_CELLS)
		
		data  = tf.placeholder(tf.float32, [None, FEATURES_PER_BAND, 1])
		target = tf.placeholder(tf.float32, [None, N_CLASSES])

		output, state = tf.nn.dynamic_rnn(network, data, dtype = tf.float32)
		
		output = tf.transpose(output, [1, 0, 2])
		last = tf.gather(output, int(output.get_shape()[0]) - 1)

		weight = tf.Variable(tf.truncated_normal([CELLS_IN_LSTM, int(target.get_shape()[1])]))
		bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

		prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
		cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

		optimiser = tf.train.AdamOptimizer()
		minimize = optimiser.minimize(cross_entropy)

		self.data_ph = data
		self.target_ph = target
		self.prediction = prediction

		# mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
		# error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

	def setTrainingData(self, filespath):

		train_data = []
		for path, dirs, files in os.walk(filespath):
			for file in [files[0]]:
				print(file)
				if ( file[len(file)-3:] == "wav" ):
					(rate, sig) = wv.read(filespath + file)
				
				train_data.append(sig.T[0])

		self.train_data_raw = train_data

		training_x = []
		training_y = []
		
		for song in train_data:
			print("Constructing song")
			for i in range(0,int((song.shape[0]-FEATURES_PER_BAND-2)/10000)):
				print("Constructing sample = " + str(i))
				training_x.append(song[ i:i+FEATURES_PER_BAND ])
				training_y.append(bf16(song[i+FEATURES_PER_BAND+1]) )

		print("Construction done")
		self.train_x = training_x
		self.train_y = training_y

	def train(self):

		print("Training commences")

		init_op = tf.initialize_all_variables()
		sess = tf.Session()
		sess.run(init_op)

		no_of_batches = int(len(self.train_x)/BATCH_SIZE)
		for i in range(EPOCH):
		    ptr = 0
		    for j in range(no_of_batches):
		        inp, out = self.train_x[ptr:ptr+BATCH_SIZE], self.train_y[ptr:ptr+BATCH_SIZE]
		        ptr += BATCH_SIZE
		        sess.run(minimize,{self.data_ph: inp, self.target_ph: out})
		    print ("Epoch - ", str(i))

		self.sess = sess

	def inventSong(self):

		print("Song gen starts")
		
		song_length = 20
		seed = np.array([ 0 for i in range(0,FEATURES_PER_BAND) ], dtype=np.int16)
		print("Song seed = ", seed)
		song = []

		fram = seed
		fram = np.reshape(fram, (-1, FEATURES_PER_BAND, 1))
		print(fram, fram.shape)

		for x in range(0,song_length * SAMPLING_RATE):
			print("iteration",x)
			next_pt = self.sess.run(self.prediction, {self.data_ph: fram})
			next_pt = [ int(x > 0.5) for x in next_pt ]
			fram_int = f16b(next_pt)
			song.append(fram_int)
			fram = np.append(fram, np.array([[[fram_int]]]), axis=1)

		print("Generating the wav file")
		wv.write("gen.wav", SAMPLING_RATE, song)
