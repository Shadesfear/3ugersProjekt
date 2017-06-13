import numpy as np
import matplotlib.pyplot as plt



l = np.linspace(350, 1100, 1100-350)
n = 1.31279 + ( 15.762 * (l**(-1)) ) - ( 4382 * (l**(-2)) ) + ( (1.1455 * (10**6)) * (l**(-3)) )

plt.figure()
plt.plot(l,n)
plt.show()
