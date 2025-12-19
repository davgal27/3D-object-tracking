import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "xyz_frame.csv")

#Load data
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)

frames = data[:, 0]
X = data[:, 1]
Y = data[:, 2]
Z = data[:, 3]

# plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Ball Trajectory (Time-lapse)")

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())

# plotting objects
trajectory, = ax.plot([], [], [], lw=2, color="blue")
point, = ax.plot([], [], [], "ro")
time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

# update animation 
def update(i):
    trajectory.set_data(X[:i+1], Y[:i+1])
    trajectory.set_3d_properties(Z[:i+1])

    point.set_data([X[i]], [Y[i]])
    point.set_3d_properties([Z[i]])

    time_text.set_text(f"Frame: {int(frames[i])}")

    return trajectory, point, time_text

# run animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(X),
    interval=30,
    blit=False 
)

plt.show()
