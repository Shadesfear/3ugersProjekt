import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fileImport import fileImport
from detect_peaks import detect_peaks
from movingaverage import movingaverage
import math
import scipy.fftpack

################################################################################
#Dict to store data
table = {}

# Set folders to import data from
cwd = os.getcwd()
dataDir = cwd + '/data/'
subdirs = [x[1] for x in os.walk(dataDir)][0]

#Importing all the datasets with the function fileImport, reference the dataset with table['file']
#No extension in file.
for subdir in subdirs:
	table.update(fileImport(dataDir + subdir,'.txt', skip_head = 19, skip_foot= 2, delimiter ='\t'))
# print(len(table))

# Save the vector I0 as the mean of the intensity without any bubbles (as a function of lambda)
I0 = [0] * 3645
for data in table:
	if 'nobub' in data:
		I0 = table[data][:,1] + I0
I0 = I0 / 50

# Find the refractive index of the solution
n1 = 1.0
n2 = 1.333
r = ((n2 - n1)/(n1 + n2)) ** 2

################################################################################
# Main code for finding lambda etc
def findlambda2(expnr, measnr):
	# Find the absorption-spectrum for experiment NMM
	if len(str(measnr)) < 2:
		exp = 'forsoeg' + str(expnr) + str(0) + str(measnr)
	else:
		exp = 'forsoeg' + str(expnr) + str(measnr) # experiment
	I = table[exp][:,1] # intensity
	a = I0 * (1 - r)**2 / (1 - r**2) - I # absorption
	A = movingaverage(a, 100, 3) # absorption (movingaverage to smooth out the function)

	# Find peaks
	peakIndices = detect_peaks(A, mph=10, mpd=100)
	peaks = []
	for peakIndex in peakIndices:
		if (table[exp][peakIndex,0] > 400 and table[exp][peakIndex,0] < 1000):
			peaks.append(table[exp][peakIndex,0])
	# print(peaks)

	# # Figure with results
	# plt.plot(table[exp][:,0], a, 'b-', label='fit')
	# plt.plot(table[exp][:,0], A, 'r-', label='fit')
	# for peak in peaks:
	# 	plt.axvline(x=peak)
	# plt.show()
	N = len(table[exp][:,0])
	T = (table[exp][-1,0] - table[exp][0,0]) / N # spacing between points (lambdas)
	freqs = np.linspace(0.0, 1.0 / (2.0 * T), N / 2.0)
	A_fourier = scipy.fftpack.fft(A)
	a_fourier = scipy.fftpack.fft(a)

	# plt.plot(freqs, 2.0/N * np.abs(A_fourier[:N//2]), 'b.')
	# plt.show()

	# maxfreqIndex = a_fourier.index(max(a_fourier[2:]))
	maxfreqIndex = np.argmax(A_fourier[2:len(A_fourier)//2])
	maxfreq = freqs[maxfreqIndex]
	# print(maxfreq)

	# Find lambda2 (inside the film) in [nm]
	# if len(peaks) > 1:
	# 	lambda1 = (max(peaks) - min(peaks)) / ( len(peaks) - 1)
	# 	lambda2 = lambda1 * ( n1 / n2 )
	# else:
	# 	lambda2 = math.nan

	lambda1 = (table[exp][-1,0] - table[exp][0,0]) / maxfreqIndex # missing a factor of 2 cf def. of freqs?
	lambda2 = lambda1 * (n1 / n2)

	return lambda2

################################################################################
lambda2 = {}
for i in range(1, 7+1):
	lambda2[str(i)] = []
	for j in range(0,99+1):
		# Following line filters out the beginning measurements (where the beaker was blocking the signal)
		if not ((i == 1 and j < 7) or (i == 2 and j < 16) or (i == 3 and j < 12) or (i == 4 and j < 9)
		or (i == 5 and j < 12) or (i == 6 and j < 11) or (i == 7 and j < 12)):
			lambda2[str(i)].append(findlambda2(i,j))
		else:
			lambda2[str(i)].append(math.nan)
		j += 1
	i += 1

time = np.linspace(0, 8, num=100)
for i in range(1, len(lambda2)+1):
	plt.plot(time, lambda2[str(i)], 'b.', label='fit')
	plt.show()
