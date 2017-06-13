import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from fileImport import fileImport
from detect_peaks import detect_peaks
from movingaverage import movingaverage
import math
import scipy.fftpack
from optparse import OptionParser
from scipy.optimize import curve_fit

################################################################################
# The method to use
parser = OptionParser()
parser.add_option("-f", "--fourier", dest="fouriermethod", help="use Fourier method", action='store_true', default=False)
parser.add_option("-g", "--graphs", dest="showgraphs", help="show all graphs", action='store_true', default=False)
(options, args) = parser.parse_args()
fouriermethod = options.fouriermethod
showgraphs = options.showgraphs

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
n2 = 1.31279 + ( 15.762 * (table['forsoeg100'][:,0]**(-1)) ) - ( 4382 * (table['forsoeg100'][:,0]**(-2)) ) + ( (1.1455 * (10**6)) * (table['forsoeg100'][:,0]**(-3)) )
r = ((n2 - n1)/(n1 + n2)) ** 2
def nindex(l):
	return 1.31279 + ( 15.762 * (l**(-1)) ) - ( 4382 * (l**(-2)) ) + ( (1.1455 * (10**6)) * (l**(-3)) )

################################################################################
# Main code for finding lambda etc
def findD(expnr, measnr, showgraphs, fouriermethod): # type = 'fourier' or 'peaks'
	# Find the absorption-spectrum for experiment NMM
	if len(str(measnr)) < 2:
		exp = 'forsoeg' + str(expnr) + str(0) + str(measnr)
	else:
		exp = 'forsoeg' + str(expnr) + str(measnr) # experiment
	I = table[exp][:,1] # intensity
	a = I0 * (1 - r)**2 / (1 - r**2) - I # absorption
	A = movingaverage(a, 100, 3) # absorption (movingaverage to smooth out the function)

	if not fouriermethod:
		# Find peaks
		peakIndices = detect_peaks(A, mph=10, mpd=100)
		peaks = []
		for peakIndex in peakIndices:
			if (table[exp][peakIndex,0] > 400 and table[exp][peakIndex,0] < 1000):
				peaks.append(table[exp][peakIndex,0])
		# print(peaks)
		if showgraphs:
			# Figure with results
			plt.figure()
			plt.title('Experiment nr: ' + str(expnr) + ' Measurement nr: ' + str(measnr))
			plt.plot(table[exp][:,0], a, 'b-', label='fit')
			plt.plot(table[exp][:,0], A, 'r-', label='fit')
			for peak in peaks:
				plt.axvline(x=peak, color='g')
			plt.show()
		# Find lambda2 (inside the film) in [nm]
		lambda2 = 0
		if len(peaks) > 1:
			for i in range(0, len(peaks) -1):
				m = (peaks[i+1]/nindex(peaks[i+1]) + peaks[i]/nindex(peaks[i])) / (2* (peaks[i+1]/nindex(peaks[i+1]) - peaks[i]/nindex(peaks[i])))
				lambda2 = (m / 2) * peaks[i]/nindex(peaks[i]) + lambda2
			lambda2 = lambda2 / (len(peaks) - 1)
		else:
			lambda2 = math.nan
		D = lambda2 / 2
		return D

	if fouriermethod:
		N = len(table[exp][:,0])
		T = (table[exp][-1,0] - table[exp][0,0]) / N # spacing between points (lambdas)
		freqs = np.linspace(0.0, 1.0 / (2.0 * T), N / 2.0)
		A_fourier = scipy.fftpack.fft(A)
		a_fourier = scipy.fftpack.fft(a)

		if showgraphs:
			plt.plot(freqs, 2.0/N * np.abs(A_fourier[:N//2]), 'b.')
			plt.show()

		maxfreqIndex = 2 + np.argmax(a_fourier[2:len(a_fourier)//2])
		# maxfreqIndex = 2 + np.argmax(A_fourier[2:len(A_fourier)//2])
		maxfreq = freqs[maxfreqIndex]
		# print(maxfreq)

		lambda1 = (table[exp][-1,0] - table[exp][0,0]) / maxfreqIndex # missing a factor of 2 cf def. of freqs?
		lambda2 = lambda1 * (n1 /n2)
		return lambda2

################################################################################

# Make dict of lambda2 where each entry is an array of lambda as a function of time for the given experiment
# (convert to d)
D = {}
for i in range(0, 7):
	D[str(i)] = []
	for j in range(0,100):
		# Following line filters out the beginning measurements (where the beaker was blocking the signal)
		if not ((i == 0 and j < 7) or (i == 1 and j < 16) or (i == 2 and j < 12) or (i == 3 and j < 9)
		or (i == 4 and j < 12) or (i == 5 and j < 11) or (i == 6 and j < 12)):
			D[str(i)].append(findD(i + 1, j, showgraphs, fouriermethod))
		else:
			D[str(i)].append(math.nan)
		j += 1
	i += 1

time = np.linspace(0, 8, num=100)

# Find concentrations of soap in different experiments
ml = [13.2, 22.1, 35.9, 41.8, 76.3, 84.0]
concentrations = [0, 0, 0, 0, 0, 0, 0]
for j in range(0, 6):
	for i in range(0,j+1):
		concentrations[j] += ml[i]
	concentrations[j] = concentrations[j] / (105.1 + concentrations[j])
concentrations[6] = 1

# Function to fit to (cf theory)
def func(x, a, b):
	return(1 / np.sqrt(a * x + b))
# Linear function
def linfunc(x,a,b):
	return(a * x + b)
# Transform of d, so is linear
def transmodel(my_list):
    return [ 1/x**2 for x in my_list ]

# Fitting to model
avals = [0, 0, 0, 0, 0, 0, 0]
bvals = [0, 0, 0, 0, 0, 0, 0]
for i in range(0, 7):
	x = []
	y = []
	for idx in range(0,len(D[str(i)])):
		if not 'nan' in str(D[str(i)][idx]):
			y.append(D[str(i)][idx])
			x.append(time[idx])
	popt, pcov = curve_fit(linfunc, x, transmodel(y), p0 = [1, 1])
	avals[i] = popt[0]
	bvals[i] = popt[1]

	# Plot fit and data
	plt.figure()
	plt.plot(time, transmodel(D[str(i)]), 'b.', label= str(round(100 * concentrations[i])) + ' vol%')
	plt.plot(time, linfunc(time, avals[i], bvals[i]), 'r-', label= 'fit')
	plt.ylabel('$1/d^2$ [nm$^{-2}$]')
	plt.xlabel('$t$ [s]')
	plt.legend()

print('Values of a (in $1/d^2 = a t + b$) for experiments 1-7:')
print(avals)
print('\n')
#print(bvals)

# Plot all data in one figure
plt.figure()
for i in range(0, 7):
	plt.plot(time, D[str(i)], '.', label= str(round(100 * concentrations[i])) + ' Vol-%')
plt.legend()
plt.axis([0, 8, 900, 3000])
plt.ylabel('$d$ [nm]')
plt.xlabel('$t$ [s]')
plt.show()
