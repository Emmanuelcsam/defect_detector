import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D surface plot of pixel intensities (x, y positions vs. grayscale value)

# Load pixel data: skip metadata rows so the header aligns correctly
df = pd.read_csv('/mnt/data/img63_detailed.csv', skiprows=6)

# Keep only the X, Y, and intensity columns
df = df[['X Coordinate', 'Y Coordinate', 'Grayscale Value (0-255)']]

# Pivot to form a 2D array of intensities (rows = Y, columns = X)
intensity_matrix = df.pivot(index='Y Coordinate', columns='X Coordinate', values='Grayscale Value (0-255)')
Z = intensity_matrix.values
x = intensity_matrix.columns.values
y = intensity_matrix.index.values
X, Y = np.meshgrid(x, y)

# Downsample for performance (every 5th pixel)
stride = 5
X_ds = X[::stride, ::stride]
Y_ds = Y[::stride, ::stride]
Z_ds = Z[::stride, ::stride]

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_ds, Y_ds, Z_ds)
ax.set_xlabel('X Pixel Position')
ax.set_ylabel('Y Pixel Position')
ax.set_zlabel('Pixel Intensity')
plt.tight_layout()
plt.show()
