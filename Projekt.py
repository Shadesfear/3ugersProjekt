import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fileImport import fileImport
from detect_peaks import detect_peaks


#Dict to store data
table = {}

cwd = os.getcwd()
dataDir = cwd + '/data/'
subdirs = [x[1] for x in os.walk(dataDir)][0]

#Importing all the datasets with the function fileImport, reference the dataset with table['file']
#No extension in file.
for subdir in subdirs:
	table.update(fileImport(dataDir + subdir,'.txt', skip_head = 19, skip_foot= 2, delimiter ='\t'))

# print(len(table))
I0 = [0] * 3645
for data in table:
	if 'nobub' in data:
		I0 = table[data][:,1] + I0
I0 = I0 / 50

n1 = 1.0
n2 = 1.333
r = ((n1 - n2)/(n1 + n2)) ** 2

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

newy = movingaverage(movingaverage(movingaverage(I0 * (1 - r) - table['forsoeg765'][:,1], 100), 100),100)

peaks = detect_peaks(newy, mph=100, mpd=500)

print(table['forsoeg765'][peaks,0])

plt.plot(table['forsoeg765'][:,0], I0 * (1 - r) - table['forsoeg765'][:,1], 'r-', label='fit')
plt.plot(table['forsoeg765'][:,0], newy, 'g-', label='fit')
plt.show()
