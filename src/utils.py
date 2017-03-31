import os
import numpy as np

def convertOggToWav(filespath):

	for path, dirs, files in os.walk(filespath):
		for file in files:
			print(file)
			if ( file[len(file)-3:] == "ogg" ):
				os.system("oggdec -o " + filespath + "/" + file[:-4] + ".wav " + filespath + "/" + file)

def binary_from_int16(num):
	inum = bin(np.uint32(num))
	return [int(b) for b in inum[2:]]
	