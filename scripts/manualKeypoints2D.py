import cv2
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the mesh
mesh = trimesh.load('/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-3B_gray.obj')
#mesh = trimesh.load('/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A.obj')
vertices = mesh.vertices
faces = mesh.faces
print(f"Total number of vertices: {len(vertices)}")

# Vertex indices you want to label
# For 3B
step = 1800
# For 1A
#step = 800

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
'''
STEP ONE: 3D MESH VISUALIZATION
'''
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

'''
STEP TWO: 2D MANUAL KEYPOINT DEFINING
'''

img = cv2.imread('/home/simon/Documents/MVSR Lab/mvsr/data/rgb/1.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

clicked_points = []

def on_click(event):
    if event.button == 1 and event.xdata is not None:
        clicked_points.append((event.xdata, event.ydata))
        print(f"Point: ({event.xdata:.1f}, {event.ydata:.1f})")

fig, ax = plt.subplots()
ax.imshow(img_rgb)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Click on the 2D points (same order as 3D)")
plt.show()

image_points = np.array(clicked_points, dtype=np.float32)

'''
STEP THREE: PNP
'''

object_points = points.astype(np.float32)

camera_matrix = np.array([
    [616.7415,    0.0,     324.8176],
    [0.0,       616.9197,  238.0456],
    [0.0,         0.0,       1.0   ]
], dtype=np.float32)

dist_coeffs = np.zeros(5)  # if you have distortion values, replace this

success, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)
# Convert rvec to rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Mesh center in object coordinates
mesh_center_obj = vertices.mean(axis=0).reshape(3, 1)

# Transform to camera coordinates
mesh_center_cam = R @ mesh_center_obj + tvec  # shape (3, 1)
# Print mesh centre position
print("\n=== Mesh Center in Camera Coordinates ===")
print(f"Translation (X, Y, Z): {mesh_center_cam.ravel()}")
# Print rotation matrix
print("\nRotation Matrix R:")
print(R)

'''
STEP FOUR: PROJECT OBJ ONTO ORIGINAL .PNG
'''

projected_all, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
projected_all = projected_all.reshape(-1, 2).astype(int)

img_with_faces = img.copy()

for face in faces:
    pts = projected_all[face]
    pts = pts.reshape(-1, 1, 2)  # required shape for polylines
    cv2.polylines(img_with_faces, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

plt.imshow(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
plt.title("Projected 3D Mesh (Faces)")
plt.axis("off")
plt.show()

# Calculate mesh center
mesh_center = vertices.mean(axis=0)

# Define small axis lines from center (in 3D)
axis_length = 20
axis_points_3d = np.array([
    mesh_center,  # origin
    mesh_center + [axis_length, 0, 0],  # X
    mesh_center + [0, axis_length, 0],  # Y
    mesh_center + [0, 0, axis_length],  # Z
], dtype=np.float32)

# Project to image
projected_axes, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
origin_2d = tuple(projected_axes[0].ravel().astype(int))
x_2d = tuple(projected_axes[1].ravel().astype(int))
y_2d = tuple(projected_axes[2].ravel().astype(int))
z_2d = tuple(projected_axes[3].ravel().astype(int))

# Draw coordinate axes on image
img_axes = img.copy()
cv2.arrowedLine(img_axes, origin_2d, x_2d, color=(0, 0, 255), thickness=2, tipLength=0.1)  # X - red
cv2.arrowedLine(img_axes, origin_2d, y_2d, color=(0, 255, 0), thickness=2, tipLength=0.1)  # Y - green
cv2.arrowedLine(img_axes, origin_2d, z_2d, color=(255, 0, 0), thickness=2, tipLength=0.1)  # Z - blue

# Show result
plt.imshow(cv2.cvtColor(img_axes, cv2.COLOR_BGR2RGB))
plt.title("Projected Coordinate Frame at Mesh Center")
plt.axis("off")
plt.show()