import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the mesh
mesh = trimesh.load('/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-3B_gray.obj')
vertices = mesh.vertices
faces = mesh.faces
print(f"Total number of vertices: {len(vertices)}")

# Vertex indices you want to label
step = 1800
selected_indices = np.arange(0, len(vertices), step)
points = vertices[selected_indices]

# Select those vertices
points = vertices[selected_indices]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_zlim([0, 100])

# Plot mesh faces
for face in faces:
    tri = vertices[face]
    tri = np.vstack([tri, tri[0]])  # close triangle
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color='gray', linewidth=0.3)

# Plot selected vertices
for i, pt in zip(selected_indices, points):
    ax.scatter(pt[0], pt[1], pt[2], color='r', s=40)
    ax.text(pt[0], pt[1], pt[2], f'{i}', color='black', fontsize=8)

ax.set_title("Selected Vertices from mesh.vertices")
plt.tight_layout()
plt.show()
