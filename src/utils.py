import os
import numpy as np
import mido

def convertOggToWav(filespath):

	for path, dirs, files in os.walk(filespath):
		for file in files:
			print(file)
			if ( file[len(file)-3:] == "ogg" ):
				os.system("oggdec -o " + filespath + "/" + file[:-4] + ".wav " + filespath + "/" + file)

def binary_from_int16(num):
	inum = bin(np.uint32(num))
	return [int(b) for b in inum[2:]]


lowerBound = 24
upperBound = 102

def midiToNoteMatrix(midifile):

	pattern = []
	try:
		pattern = mido.MidiFile(midifile)
	except Exception as e:
		return []

	timeleft = [track[0].time for track in pattern.tracks]
	posns = [0 for track in pattern]
	statematrix = []
	span = upperBound-lowerBound
	time = 0

	state = [[0,0] for x in range(span)]
	statematrix.append(state)
	while True:
		if time % (pattern.ticks_per_beat / 4) == (pattern.ticks_per_beat / 8):
			# Crossed a note boundary. Create a new state, defaulting to holding notes
			oldstate = state
			state = [[oldstate[x][0],0] for x in range(span)]
			statematrix.append(state)

		for i in range(len(timeleft)):
			while timeleft[i] == 0:
				track = pattern.tracks[i]
				pos = posns[i]

				evt = track[pos]
				if (evt.type == 'note_off' or evt.type == 'note_on') :
					if (evt.note < lowerBound) or (evt.note >= upperBound):
						pass
						# print "Note {} at time {} out of bounds (ignoring)".format(evt.note, time)
					else:
						if evt.type == 'note_off' or evt.velocity == 0:
							state[evt.note-lowerBound] = [0, 0]
						else:
							state[evt.note-lowerBound] = [1, 1]
				elif evt.type == 'time_signature':
					if evt.numerator not in (2, 4):
						# We don't want to worry about non-4 time signatures. Bail early!
						# print "Found time signature event {}. Bailing!".format(evt)
						return statematrix

				try:
					timeleft[i] = track[pos + 1].time
					posns[i] += 1
				except IndexError:
					timeleft[i] = None

			if timeleft[i] is not None:
				timeleft[i] -= 1

		if all(t is None for t in timeleft):
			break

		time += 1

	return [ [iden for note in state for iden in note] for state in statematrix ]

def noteMatrixToMidi(statematrix, name="example"):

	statematrix = np.asarray([ np.reshape(np.array(state), (int(len(state)/2), 2)) for state in statematrix ])
	# statematrix = np.asarray(statematrix)
	pattern = mido.MidiFile()
	track = mido.MidiTrack()
	pattern.tracks.append(track)

	# os.chdir('../media')

	print(statematrix.shape)

	default_header = (mido.MidiFile('media/ballade1.mid').tracks[0])[1:4]
	# os.chdi
	for message in default_header:
		track.append(message)
	
	span = upperBound-lowerBound
	tickscale = 55

	lastcmdtime = 0
	prevstate = [[0,0] for x in range(span)]
	for time, state in enumerate(statematrix + [prevstate[:]]):  
		offNotes = []
		onNotes = []
		for i in range(span):
			n = state[i]
			p = prevstate[i]
			if p[0] == 1:
				if n[0] == 0:
					offNotes.append(i)
				elif n[1] == 1:
					offNotes.append(i)
					onNotes.append(i)
			elif n[0] == 1:
				onNotes.append(i)
		for note in offNotes:
			track.append(mido.Message(type='note_off', time=(time-lastcmdtime)*tickscale, note=note+lowerBound))
			lastcmdtime = time
		for note in onNotes:
			track.append(mido.Message(type='note_on', time=(time-lastcmdtime)*tickscale, velocity=40, note=note+lowerBound))
			lastcmdtime = time

		prevstate = state

	eot = mido.MetaMessage(type='end_of_track', time=1)
	track.append(eot)

	pattern.save("{}.mid".format(name))