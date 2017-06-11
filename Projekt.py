import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fileImport import fileImport
from detect_peaks import detect_peaks
from movingaverage import movingaverage


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
r = ((n1 - n2)/(n1 + n2)) ** 2

# Find the absorption-spectrum for experiment NMM
exp = 'forsoeg765' # experiment
I = table[exp][:,1] # intensity
a = I0 * (1 - r) - I # absorption
A = movingaverage(movingaverage(movingaverage(movingaverage(movingaverage(a, 100), 100),100),100),100) # absorption (movingaverage to smooth out the function)

plt.plot(table[exp][:,0], I0)
plt.show()

# Find peaks
peakIndices = detect_peaks(A, mph=100, mpd=100)
peaks = []
for peakIndex in peakIndices:
	peaks.append(table[exp][peakIndex,0])
print(peaks)

# Find lambda2 (inside the filmen) in [nm]
lambda2 = (max(peaks) - min(peaks)) / len(peaks)
lambda1 = lambda2 * ( n2 / n1 )
print(lambda2)

# Figure with results
plt.plot(table[exp][:,0], a, 'b-', label='fit')
plt.plot(table[exp][:,0], A, 'r-', label='fit')
for peak in peaks:
	plt.axvline(x=peak)
plt.show()
