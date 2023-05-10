import matplotlib.pyplot as plt
import numpy as np

x_values = [0.3, 0.4, 0.6, 0.9]
y_values = [1, 1.4, 2, 3]
z_values = [44, 70, 80, 50]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(x_values, y_values, z_values,
            cmap='viridis')
plt.show()
