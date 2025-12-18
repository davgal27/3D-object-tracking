import matplotlib.pyplot as plt
import numpy as np

# Load CSV (skip header)
data = np.loadtxt("outputs/xyz_frame.csv", delimiter=',', skiprows=1)

frames = data[:, 0]
X = data[:, 1]
Y = data[:, 2]
Z = data[:, 3]

# Normalize frame numbers for color mapping
colors = (frames - frames.min()) / (frames.max() - frames.min())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each segment with a color according to time
for i in range(len(X)-1):
    ax.plot(X[i:i+2], Y[i:i+2], Z[i:i+2], color=plt.cm.viridis(colors[i]))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory Colored by Time')

# Optional: add colorbar to indicate time progression
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=frames.min(), vmax=frames.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Frame number')

plt.show()
