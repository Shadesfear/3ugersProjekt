import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fileImport import fileImport

#Dict to store data
table ={}

cwd = os.getcwd()
dataDir = cwd+'\\data\\'
subdirs = [x[1] for x in os.walk(dataDir)][0]


#Importing all the datasets with the function fileImport, reference the dataset with table['file']
#No extension in file.
for x in subdirs:
	table.update(fileImport(dataDir + x,'.txt', skip_head = 19, skip_foot= 2, delimiter ='\t'))

arb = {}
ref = ((1-1.33)**2)/(2.33)**2
transm = (1-ref)**2/(1-ref**2)

print(transm)
for x in table:
	arb[x] = table['nobub165'][:,1]*transm - table[x][:,1]

print(arb['forsoeg464'])

def movingaverage (values, window):
	weights = np.repeat(1.0, window)/window
	sma = np.convolve(values, weights, 'valid')
	return sma

def movingaveragex(values, window, N):
	va = values
	for x in range(0,N,1):
		ma = movingaverage(va, window)
		va = ma
	return ma

wi= 40
Na = 3
for x in range(1,8,1):
	plt.figure(x)
	plt.plot(table['forsoeg456'][:-(wi-1)*Na,0],movingaveragex(arb['forsoeg'+str(x)+'20'],wi,Na),"s",table['forsoeg456'][:-(wi-1)*Na,0],movingaveragex(arb['forsoeg'+str(x)+'40'],wi,Na),"*",table['forsoeg456'][:-(wi-1)*Na,0],movingaveragex(arb['forsoeg'+str(x)+'60'],wi,Na),"x",table['forsoeg456'][:-(wi-1)*Na,0],movingaveragex(arb['forsoeg'+str(x)+'80'],wi,Na),"o")

plt.legend()

plt.show()