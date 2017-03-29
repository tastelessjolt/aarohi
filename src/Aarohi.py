import tensorflow as tf
import numpy as np
import wave
import os

CELLS_IN_LSTM = 24

class Aarohi():
	"""docstring for Aarohi"""
	def __init__(self):

		# lstm = tf.contrib.rnn.BasicLSTMCell(num_units = CELLS_IN_LSTM)
		
	
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



		pass

	def inventSong(self):
		pass