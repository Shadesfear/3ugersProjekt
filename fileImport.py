import os
import numpy as np

def fileImport(directory, ext ,delimiter,skip_head=0,skip_foot=0):
	#global table
	table= {}
	files = os.listdir(directory)

	if skip_head is None:
		skip_header = 0
	if skip_head is None:
		skip_footer = 0

	for x in files:
		if ext in x:
			table[x[:-len(ext)]] = np.genfromtxt(directory+'\\'+x, delimiter = delimiter, skip_header = skip_head, skip_footer = skip_foot)
	print('All Files in ' + directory + ' has been imported')
	return table
