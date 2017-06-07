import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fileImport import fileImport

table ={}

cwd = os.getcwd()
subdirs = [x[1] for x in os.walk(cwd+'\data')][0]

for x in subdirs:
	table.update(fileImport(os.getcwd()+'\\data\\'+x,'.txt', skip_head = 19, skip_foot= 2, delimiter ='\t'))

print(len(table))