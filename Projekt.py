import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def fileImport(directory, deli, ext, skip_head, skip_foot):
	global table
	table= {}
	files = os.listdir(directory)

	if skip_head is None:
		skip_header = 0
	if skip_head is None:
		skip_footer = 0

	for x in files:
		if ext in x:
			table[x[:-len(ext)]] = np.genfromtxt(directory+'\\'+x, delimiter = deli, skip_header = skip_head, skip_footer = skip_foot)
	print('All Files in ' + directory + ' has been imported')



fileImport(os.getcwd()+'\\data\\baggrund','\t','.txt', 19, 2)
print(table["nobub160"][1])


'''

df = {}		#df er et dictionary der indeholder alt data
files = []
files2 = []
cwd = os.getcwd()
subdirs = [x[1] for x in os.walk(cwd+'\data')][0]


files.extend([os.listdir(cwd+'\data\\'+x) for x in subdirs])

k=0
for x in files:			#x er arrays med navne på vores txt filer, hvert x hører til 1 forsøg	
	for i in x:
		files2.append(i)
		df[i[:-4]] = np.genfromtxt(cwd+'\data\\'+subdirs[k]+'\\'+i, delimiter='\t', skip_header = 19, skip_footer = 1)

	k = k + 1

print(files2[500])
#print(df["forsoeg101"])
'''