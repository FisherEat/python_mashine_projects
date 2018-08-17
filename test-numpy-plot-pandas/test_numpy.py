
# a = np.array([1,2,3])
# print (a)
#
# a = np.array([2,3,4], dtype=complex)
#
# print(a)

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,3*np.pi, 0.1)

y = np.sin(x)

plt.title('sine wave form ')

plt.plot(x, y)

plt.show()

