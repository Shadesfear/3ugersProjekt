import os
import numpy as np

def fileImport(directory, ext ,delimiter,skip_head=0,skip_foot=0):
	#global table
	table= {}
	files = os.listdir(directory)
	nanCounter = 0
	if skip_head is None:
		skip_header = 0
	if skip_head is None:
		skip_footer = 0

	for x in files:
		if ext in x:
			try:
				table[x[:-len(ext)]] = np.genfromtxt(directory+'\\'+x, delimiter = delimiter, skip_header = skip_head, skip_footer = skip_foot)
			except:
				print('Some error occured while doing stuff to '  + x)
				
		try:		
			if 'nan' in str(table[x[:-len(ext)]]):
				print('Something is returning nan in ' + x)
				nanCounter += 1
		except KeyError:
			print('KeyError ' + x)

	if len(table) == len(files) and nanCounter == 0 :
		print('All functions in ' + directory + ' has been imported')


	return table
