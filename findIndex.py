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


################################################################################
#Dict to store data
table = {}

# Set folders to import data from
cwd = os.getcwd()
dataDir = cwd + '/dataindex/'

#Importing all the datasets with the function fileImport, reference the dataset with table['file']
#No extension in file.
table.update(fileImport(dataDir,'.txt', skip_head = 19, skip_foot= 2, delimiter ='\t'))
# print('Length of table: '+  str(len(table)))

d = {}
d['empty'], d['0'], d['25'], d['50'], d['75'], d['100'] = [0] * 3645, [0] * 3645, [0] * 3645, [0] * 3645, [0] * 3645, [0] * 3645

for data in table:
	if int(data[len(data)-2:len(data)]) < 20:
		d['empty'] = table[data][:,1] + d['empty']
	elif int(data[len(data)-2:len(data)]) < 30:
		d['100'] = table[data][:,1] + d['100']
	elif int(data[len(data)-2:len(data)]) < 40:
		d['25'] = table[data][:,1] + d['25']
	elif int(data[len(data)-2:len(data)]) < 50:
		d['50'] = table[data][:,1] + d['50']
	elif int(data[len(data)-2:len(data)]) < 60:
		d['75'] = table[data][:,1] + d['75']
	elif int(data[len(data)-2:len(data)]) < 70:
		d['0'] = table[data][:,1] + d['0']

for data in d:
	if data = empty:
		d[data] = d[data] / 20
	else:
		d[data] = d[data] / 10
n1 = 1.0

# Find n2 using formulars (I = I0 * (1 -r)^2 / (1 - r^2) and r = (n2 - n1)^2 / (n1 +  n2)^2)
# and using the difference in I from when cuvette is empty and containing only demin. water



# Find n2 using formulars (I = I0 * (1 -r)^2 / (1 - r^2) and r = (n2 - n1)^2 / (n1 +  n2)^2)
# and using the difference in I from when cuvette is empty and containing different concetrations
