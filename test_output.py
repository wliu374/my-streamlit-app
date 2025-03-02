import numpy as np
import matplotlib.pyplot as plt

output_map = np.load("output_map.npy")
plt.imshow(output_map, cmap='gray')
plt.show()

