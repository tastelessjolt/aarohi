import keras
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

BATCH_SIZE = 100

INPUT_SIZE = 15
MESSAGE_SIZE = 24

N_CLASSES = 16
EPOCH = 4

class Aarohi():
	"""docstring for Aarohi"""
	def __init__(self):

		self.model = Sequential()

		self.model.add(Dense(MESSAGE_SIZE, input_shape=(INPUT_SIZE, MESSAGE_SIZE), activation='relu'))
		self.model.add(LSTM(MESSAGE_SIZE))
		self.model.add(Dense(MESSAGE_SIZE, activation='relu'))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		self.model.summary()

	def load_model(self, filename):

		self.model = keras.models.load_model(filename)


	def setTrainingData(self, filespath):

		train_data = []

		for path, dirs, files in os.walk(filespath):
			for file in files:
				if file[-4:] != ".mid":
					continue
				dat_array = []
				try:
					dat = mido.MidiFile(filespath + file)
				except Exception as e:
					continue
				for i, track in enumerate(dat.tracks):
					msgs = [ msg for msg in track if not msg.is_meta ]
					if msgs:
						msgs.sort(key=lambda message: message.time)
						msgs = [ [int(l) for l in list(bt(hex=msg.hex()))] for msg in msgs]
						msgs = [ msg+[0]*(MESSAGE_SIZE - len(msg)) for msg in msgs]
						train_data.extend( msgs)
						# print([int(l) for l in list(bt(hex=msgs[0].hex()))])


		x_train = np.zeros((len(train_data)-INPUT_SIZE-1, INPUT_SIZE, 24))
		y_train = np.zeros((len(train_data)-INPUT_SIZE-1, 24))

		print(len(train_data)-INPUT_SIZE-1)
		for i in range(0,len(train_data)-INPUT_SIZE-1):
			temp = np.array(train_data[i:i+INPUT_SIZE])
			if(temp.size == INPUT_SIZE * 24):
				x_train[i, :, :] = temp
			temp = np.array(train_data[i+INPUT_SIZE+1])
			if(temp.size == 24):
				y_train[i, :] = temp
		
		self.x_train = x_train
		self.y_train = y_train
		print(self.x_train.shape)
		print(self.y_train.shape)

	def train(self):

		self.model.fit(self.x_train, self.y_train, epochs=EPOCH, batch_size=BATCH_SIZE)

	def save_model(self, filename):

		self.model.save(filename)

	def inventSong(self):

		song_length = 5000
		seed = [bytearray(b'\x90$d'), bytearray(b'\x90B\r'), bytearray(b'\x90AC'), bytearray(b'\xb0@\x7f'), bytearray(b'\x80+@'), bytearray(b'\x90>\x13'), bytearray(b'\x80>@'), bytearray(b'\xb0@\x7f'), bytearray(b'\x909\n'), bytearray(b'\x90E5'), bytearray(b'\x90+\x1f'), bytearray(b'\x90OU'), bytearray(b'\x90F\x0f'), bytearray(b'\x80F@'), bytearray(b'\x80A@')]

		seed = [ [int(l) for l in list(bt(hex=msg.hex()))] for msg in seed]
		seed = [ msg+[0]*(MESSAGE_SIZE - len(msg)) for msg in seed]
		# seed = list(map())
		seed = np.array(seed)
		song = seed
		for i in range(song_length):
			prediction = self.model.predict(np.reshape(song[i:i+INPUT_SIZE], (-1, INPUT_SIZE, 24) ) )
			song = np.concatenate((song, prediction), axis=0)

		return song
		# song_bytes = [bt(msg).tobytes() for msg in song]

