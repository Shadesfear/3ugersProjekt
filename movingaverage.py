import numpy as np

# Function for finding the
def movingaveragePre(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def movingaverage(interval, window_size, N):
    y = interval
    for i in range(1, N + 1):
        y = movingaveragePre(y, window_size)
        i += 1
    return y
