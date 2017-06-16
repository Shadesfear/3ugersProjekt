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
import matplotlib.ticker as mtick


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
		peakIndices = detect_peaks(A, mph=-1000, mpd=100)
		peaks = []
		for peakIndex in peakIndices:
			if (table[exp][peakIndex,0] > 500 and table[exp][peakIndex,0] < 900):
				peaks.append(table[exp][peakIndex,0])
		dipIndices = detect_peaks(-A, mph=-1000, mpd=100)
		dips = []
		for dipIndex in dipIndices:
			if (table[exp][dipIndex,0] > 500 and table[exp][dipIndex,0] < 900):
				dips.append(table[exp][dipIndex,0])
		# print(peaks)
		if showgraphs and ((expnr == 1) and (measnr == 20)):
			# Figure with results
			plt.figure(figsize=(3.5, 2.8))
			# plt.title('Experiment nr: ' + str(expnr) + ' Measurement nr: ' + str(measnr))
			plt.plot(table[exp][:,0], a, 'b-', label='fit', linewidth=0.7)
			plt.plot(table[exp][:,0], A, 'r-', label='fit', linewidth=2.1)
			for peak in peaks:
				plt.axvline(x=peak, color='k', ls='dashed', linewidth=1.4)
			# for dip in dips:
			# 	plt.axvlifFne(x=dip, color='k', ls='dashdot')
			plt.xlabel('$\lambda$ [nm]', fontsize=8)
			plt.ylabel('$\\frac{(1 - r)^2}{1 - r^2} I_0 - I$ [n]', fontsize=8)
			plt.xticks(fontsize = 6)
			plt.yticks(fontsize = 6, rotation='vertical')
			plt.tight_layout()
			plt.savefig('absorbance.png', dpi=600)
			print('Figure saved!')

		# Find D of the film in [nm]
		Dp = []
		Dd = []

		if len(peaks) > 1:
			for i in range(0, len(peaks) -1):
				m = (peaks[i+1]/nindex(peaks[i+1]) + peaks[i]/nindex(peaks[i])) / (2* (peaks[i+1]/nindex(peaks[i+1]) - peaks[i]/nindex(peaks[i])))
				Dp.append((nindex(peaks[i]) / 2) * peaks[i]/nindex(peaks[i]) * (m + 1/2) )

		if len(dips) > 1:
			for i in range(0, len(dips) -1):
				m = (dips[i+1]/nindex(dips[i+1])) / ((dips[i+1]/nindex(dips[i+1]) - dips[i]/nindex(dips[i])))
				Dd.append( (nindex(dips[i]) / 2) * dips[i]/nindex(dips[i]) * m  )

		if(Dp and Dd):
			D = (1/2 * abs(np.mean(Dp))  + 1/2 * abs(np.mean(Dd)))
			sigD = np.sqrt(1/4 * (np.std(Dp)**2) + 1/4 * (np.std(Dd)**2))
		elif Dp:
				D = np.mean(Dp)
				sigD = np.std(Dp)
		else:
			D = math.nan
			sigD = math.nan
		return D, sigD

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
sigD = {}
for i in range(0, 7):
	D[str(i)] = []
	sigD[str(i)] = []
	for j in range(0,100):
		# Following line filters out the beginning measurements (where the beaker was blocking the signal)
		if not ((i == 0 and j < 7) or (i == 1 and j < 16) or (i == 2 and j < 12) or (i == 3 and j < 9)
		or (i == 4 and j < 12) or (i == 5 and j < 11) or (i == 6 and j < 12)):
			x, sigx = findD(i + 1, j, showgraphs, fouriermethod)
			D[str(i)].append(x)
			sigD[str(i)].append(sigx)
		else:
			D[str(i)].append(math.nan)
			sigD[str(i)].append(math.nan)
		j += 1
	i += 1

time = np.linspace(0, 8, num=100)

# Find concentrations of soap in different experiments
ml = [13.2, 22.1, 35.9, 41.8, 76.3, 84.0]
totml = [0, 0, 0, 0, 0, 0, 0]
concentrations = [0, 0, 0, 0, 0, 0, 0]
for j in range(0, 6):
	for i in range(0,j+1):
		concentrations[j] += ml[i]
		totml[j] += ml[i]
	concentrations[j] = concentrations[j] / (105.1 + concentrations[j])
concentrations[6] = 1

# Function to fit to (cf theory)
def func(x, a, b):
	return(1 / np.sqrt(a * x + b))
# Linear function
def linfunc(x,a,b):
	return(a * x + b)
def linfuncarray(my_list,a,b):
	return [ linfunc(x, a, b) for x in my_list ]
# Transform of d, so is linear
def transmodel(my_list):
    return [ 1/x**2 for x in my_list ]

def times(list1, list2):
	return([list1[i]*list2[i] for i in range(len(list1))])

# Fitting to model
avals = [0, 0, 0, 0, 0, 0, 0]
bvals = [0, 0, 0, 0, 0, 0, 0]
asig = [0, 0, 0, 0, 0, 0, 0]
bsig = [0, 0, 0, 0, 0, 0, 0]
cov = [0, 0, 0, 0, 0, 0, 0]
for i in range(0, 7):
	x1, y1, sigy1 = [], [], []
	x, y, sigy = [], [], []
	# Remove NaNs
	for idx in range(0,len(D[str(i)])):
		if not 'nan' in str(D[str(i)][idx]):
			y1.append(1 / (D[str(i)][idx] ** 2) )
			x1.append(time[idx])
			sigy1.append((2 / (D[str(i)][idx] ** 3) ) * sigD[str(i)][idx])

	# Remove outliers and only choose first second
	for idx in range(0,len(x1)):
		if x1[idx] < (x1[0] + 1):
			if (i != 3) or (i == 3 and x1[idx] > 1.1):
				x.append(x1[idx])
				y.append(y1[idx])
				sigy.append(sigy1[idx])

	# print(str(len(x1)) + '  ' + str(len(y1)) + '  ' + str(len(sigy1)))
	popt, pcov = curve_fit(linfunc, x, y, sigma=sigy)
	avals[i] = popt[0]
	bvals[i] = popt[1]
	asig[i] = np.sqrt(pcov[0][0])
	bsig[i] = np.sqrt(pcov[1][1])
	cov[i] = pcov[0][1]

	# Plot fit and data
	fig, ax = plt.subplots(figsize=(3.5, 2.8))
	ax.errorbar(x1, y1, yerr=sigy1, fmt='b.', label= str(round(100 * concentrations[i])) + ' vol%', markersize=1.1, linewidth=0.7)
	plt.plot(time, linfunc(time, avals[i], bvals[i]), 'r-', label= 'fit', linewidth=0.7)
	plt.xlim(0.8, 8.1)
	plt.ylim(0, 0.0000004)
	plt.ylabel('$1/d^2$ [nm$^{-2}$]', fontsize=8)
	plt.xlabel('$t$ [s]', fontsize=8)
	plt.xticks(fontsize = 6)
	plt.yticks(fontsize = 6)
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
	plt.tight_layout()
	plt.savefig('1overd2_' + str(i) + '.png', dpi=600)

	chi2 = 0
	for i in range(len(x)):
		chi2 = (((popt[0] * x[i] + popt[1]) - y[i])**2)/((sigy[i])**2) + chi2
	chi2 = chi2 / (len(x) - 1)
	# print(chi2)
print('\n Values of a (in $1/d^2 = a t + b$) for experiments 1-7:')
print(avals)
print(asig)
#print(bvals)

def times100(array):
	return [ 100 * x for x in array ]

x = [times100(concentrations)[i] for i in [0, 1, 2, 4, 5, 6]]
y = [avals[i] for i in [0, 1, 2, 4, 5, 6]]
sigy = [asig[i] for i in [0, 1, 2, 4, 5, 6]]
sigxm = [(100 * 0.1 * np.sqrt(((totml[i])/((totml[i] + 105.1)**2))**2 + (1/(totml[i] + 105.1))**2)) for i in range(7)]
sigx = [sigxm[i] for i in [0, 1, 2, 4, 5, 6]]
popt, pcov = curve_fit(linfunc, x, y, sigma=sigy)

fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.errorbar(x, y, yerr=sigy, xerr=sigx, fmt='.', markersize=5.0)
plt.plot([0, 110], [(popt[1]), (110*popt[0] + popt[1])], 'r--', linewidth=0.7)
plt.xlim(10, 105)
plt.xlabel('$C($soap$)$ [%]', fontsize=8)
plt.ylabel('$a$ [1/s$\cdot$nm$^2$]', fontsize=8)
plt.xticks(fontsize = 6)
plt.yticks(fontsize = 6)
plt.tight_layout()
plt.savefig('a.png', dpi=600)

# Plot all data in one figure
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for i in [0,1,2,4,5,6]:
	ax.errorbar(time, D[str(i)], sigD[str(i)], fmt='.', label= str(round(100 * concentrations[i])) + ' Vol-%', markersize=1.1, linewidth=0.7)
plt.ylabel('$d$ [nm]', fontsize=8)
plt.xlabel('$t$ [s]', fontsize=8)
plt.xlim(0.5, 8.1)
plt.ylim(1287, 4525)
plt.xticks(fontsize = 6)
plt.yticks(fontsize = 6, rotation='vertical')
plt.tight_layout()
plt.savefig('dmeasured.png', dpi=600)
plt.show()
