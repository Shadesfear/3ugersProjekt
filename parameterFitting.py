import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from fileImport import fileImport
from scipy.optimize import curve_fit

pi = math.pi

def intensity(l, d):
    n1 = 1.0
    n2 = 1.333
    r = ((n1 - n2)/(n1 + n2)) ** 2
    return (1 - r) / (1 + r - (2 * r * np.cos(4 * pi * n2 * d / l))) - 1
