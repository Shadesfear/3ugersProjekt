import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = {}		#df er et dictionary der indeholder alt data
files = []
cwd = os.getcwd()
subdirs = [x[1] for x in os.walk(cwd+'\data')][0]


files.extend([os.listdir(cwd+'\data\\'+x) for x in subdirs])

k=0
for x in files:			#x er arrays med navne på vores txt filer, hvert x hører til 1 forsøg
	
	for i in x:
		df[i[:-4]] = np.genfromtxt(cwd+'\data\\'+subdirs[k]+'\\'+i, delimiter='\t', skip_header = 19, skip_footer = 1)

	k = k + 1


print(df["forsoeg101"])