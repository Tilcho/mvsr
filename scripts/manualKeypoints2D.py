import cv2
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the mesh
# For model Achse-3B:
# mesh = trimesh.load('/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-3B_gray.obj')
# For model Achse-1A:
mesh = trimesh.load('/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj')
vertices = mesh.vertices
faces = mesh.faces
print(f"Total number of vertices: {len(vertices)}")

# Vertex selection step size (determines how many points are labeled)
# Use different step for different models
step = 800  # for Achse-1A
selected_indices = np.arange(0, len(vertices), step)
points = vertices[selected_indices]

# Plot the mesh and highlight selected vertices
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
ax.set_zlim([0, 100])

'''
STEP ONE: 3D MESH VISUALIZATION
'''
# Draw each triangle in the mesh
for face in faces:
    tri = vertices[face]
    tri = np.vstack([tri, tri[0]])  # Close the triangle loop
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color='gray', linewidth=0.3)

# Highlight and label selected vertices
for i, pt in zip(selected_indices, points):
    ax.scatter(pt[0], pt[1], pt[2], color='r', s=40)
    ax.text(pt[0], pt[1], pt[2], f'{i}', color='black', fontsize=8)

ax.set_title("Selected Vertices from mesh.vertices")
plt.tight_layout()
plt.show()

'''
STEP TWO: 2D MANUAL KEYPOINT DEFINING
'''

# Load and display the reference RGB image for 2D point annotation
img = cv2.imread('/home/simon/Documents/MVSR Lab/mvsr/data/rgb/1.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

clicked_points = []

# Define click handler to collect 2D image points
def on_click(event):
    if event.button == 1 and event.xdata is not None:
        clicked_points.append((event.xdata, event.ydata))
        print(f"Point: ({event.xdata:.1f}, {event.ydata:.1f})")

# Show image and collect points via mouse clicks
fig, ax = plt.subplots()
ax.imshow(img_rgb)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Click on the 2D points (same order as 3D)")
plt.show()

# Convert to NumPy array
image_points = np.array(clicked_points, dtype=np.float32)

'''
STEP THREE: PNP
'''

# Object points (3D) corresponding to clicked image points
object_points = points.astype(np.float32)

# Camera intrinsic matrix
camera_matrix = np.array([
    [616.7415,    0.0,     324.8176],
    [0.0,       616.9197,  238.0456],
    [0.0,         0.0,       1.0   ]
], dtype=np.float32)

# No lens distortion
dist_coeffs = np.zeros(5)

# Solve PnP to get rotation and translation from object to camera
success, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

# Convert rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Compute mesh center in object coordinates
mesh_center_obj = vertices.mean(axis=0).reshape(3, 1)

# Transform center to camera coordinates
mesh_center_cam = R @ mesh_center_obj + tvec

# Print center and rotation matrix
print("\n=== Mesh Center in Camera Coordinates ===")
print(f"Translation (X, Y, Z): {mesh_center_cam.ravel()}")
print("\nRotation Matrix R:")
print(R)

'''
STEP FOUR: PROJECT OBJ ONTO ORIGINAL .PNG
'''

# Project all mesh vertices to 2D image space
projected_all, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
projected_all = projected_all.reshape(-1, 2).astype(int)

# Copy image and draw projected mesh faces
img_with_faces = img.copy()
for face in faces:
    pts = projected_all[face]
    pts = pts.reshape(-1, 1, 2)  # Format for OpenCV drawing
    cv2.polylines(img_with_faces, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

# Show the image with projected mesh
plt.imshow(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
plt.title("Projected 3D Mesh (Faces)")
plt.axis("off")
plt.show()

# Compute mesh center
mesh_center = vertices.mean(axis=0)

# Define 3D axes centered at mesh center
axis_length = 20
axis_points_3d = np.array([
    mesh_center,  # origin
    mesh_center + [axis_length, 0, 0],  # X-axis
    mesh_center + [0, axis_length, 0],  # Y-axis
    mesh_center + [0, 0, axis_length],  # Z-axis
], dtype=np.float32)

# Project 3D axes to 2D image
projected_axes, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
origin_2d = tuple(projected_axes[0].ravel().astype(int))
x_2d = tuple(projected_axes[1].ravel().astype(int))
y_2d = tuple(projected_axes[2].ravel().astype(int))
z_2d = tuple(projected_axes[3].ravel().astype(int))

# Draw the axes on the image
img_axes = img.copy()
cv2.arrowedLine(img_axes, origin_2d, x_2d, color=(0, 0, 255), thickness=2, tipLength=0.1)  # X - red
cv2.arrowedLine(img_axes, origin_2d, y_2d, color=(0, 255, 0), thickness=2, tipLength=0.1)  # Y - green
cv2.arrowedLine(img_axes, origin_2d, z_2d, color=(255, 0, 0), thickness=2, tipLength=0.1)  # Z - blue

# Show final image with coordinate axes
plt.imshow(cv2.cvtColor(img_axes, cv2.COLOR_BGR2RGB))
plt.title("Projected Coordinate Frame at Mesh Center")
plt.axis("off")
plt.show()
