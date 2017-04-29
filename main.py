from src.Aarohi import Aarohi

import numpy as np
import sys
import signal

aarohi = Aarohi() # our trainer and predictor ! :P 
aarohi.setTrainingData("./media/") # takes the training data in .mid format from the folder path given
aarohi.load_model("model9.h5") # this was our best model
def signal_handler(signal, frame): # to capture Ctrl + C and save the model middle way of training
	aarohi.save_model("model10.h5")
	print('You pressed Ctrl+C!')
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

aarohi.train() # actual training
aarohi.save_model("model10.h5") # And now after we trained save the model! 
aarohi.inventSong('') # Finally get the song out of the model with some seed  