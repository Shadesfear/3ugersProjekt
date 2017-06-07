import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fileImport import fileImport

#Dict to store data
table ={}

cwd = os.getcwd()
subdirs = [x[1] for x in os.walk(cwd+'\data')][0]


#Importing all the datasets with the function fileImport, reference the dataset with table['file']
#No extension in file.
for x in subdirs:
	table.update(fileImport(os.getcwd()+'\\data\\' + x,'.txt', skip_head = 19, skip_foot= 2, delimiter ='\t'))

