import os

def convertOggToWav(filespath):

	for path, dirs, files in os.walk(filespath):
		for file in files:
			print(file)
			if ( file[len(file)-3:] == "ogg" ):
				os.system("oggdec -o " + filespath + "/" + file[:-4] + ".wav " + filespath + "/" + file)
