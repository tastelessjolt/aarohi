from keras.models import Sequential
from keras.layers import Dense, LSTM

from bitstring import BitArray as bt

import numpy as np
import pandas as pd
import mido
import os

np.random.seed(100)

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

BATCH_SIZE = 10

INPUT_SIZE = 15
MESSAGE_SIZE = 24

N_CLASSES = 16
EPOCH = 8

class Aarohi():
	"""docstring for Aarohi"""
	def __init__(self):

		self.model = Sequential()

		self.model.add(LSTM(MESSAGE_SIZE*2, input_shape=(INPUT_SIZE, MESSAGE_SIZE)))
		self.model.add(Dense(MESSAGE_SIZE, activation='relu'))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		self.model.summary()

	def load_model(self, filename):

		self.model = keras.models.load_model(filename)


	def setTrainingData(self, filespath):

		train_data = []

		for path, dirs, files in os.walk(filespath):
			for file in files[:5]:
				if file[-4:] != ".mid":
					continue
				dat_array = []
				dat = mido.MidiFile(filespath + file)
				for i, track in enumerate(dat.tracks):
					msgs = [ msg for msg in track[:10] if not msg.is_meta ]
					if msgs:
						msgs.sort(key=lambda message: message.time)
						msgs = [ [int(l) for l in list(bt(hex=msg.hex()))] for msg in msgs]
						msgs = [ msg+[0]*(MESSAGE_SIZE - len(msg)) for msg in msgs]
						train_data.extend( msgs)
						# print([int(l) for l in list(bt(hex=msgs[0].hex()))])

		x_train = []
		y_train = []

		for i in range(0,len(train_data)-INPUT_SIZE-1):
			x_train.append(train_data[i:i+INPUT_SIZE])
			y_train.append(train_data[i+INPUT_SIZE+1])
		
		self.x_train = np.array(x_train)
		self.y_train = np.array(y_train)
		# print(self.x_train)
		# print(self.y_train)

	def train(self):

		self.model.fit(self.x_train, self.y_train, epochs=EPOCH, batch_size=BATCH_SIZE)

	def save_model(self, filename):

		self.model.save(filename)

	def inventSong(self):

		song_length = 20
		# seed = np.array([ 0 for i in range(0,FEATURES_PER_BAND) ], dtype=np.int16)
		seed = np.random.rand(3 * FEATURES_PER_BAND) + 0.5
		seed[seed < 0] = 0
		print("Song seed = ", seed)
		song = []

		fram = seed
		fram = np.reshape(fram, (-1, FEATURES_PER_BAND, 1))
		print(fram, fram.shape)

		for x in range(0,song_length * SAMPLING_RATE):
			print("iteration",x)
			next_pt = self.sess.run(self.prediction, {self.data_ph: fram})
			print(next_pt)
			next_pt = [ int(x > 0.03) for x in next_pt[0] ]
			fram_int = f16b(next_pt)
			song.append(fram_int)
			fram = np.append(np.array([fram[0][1:]]), np.array([[[fram_int]]]), axis=1)

		print("Generating the wav file")
		wv.write("gen.wav", SAMPLING_RATE, song)
