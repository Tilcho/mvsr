import open3d as o3d
import numpy as np
import cv2
import os
from math import radians, sin, cos

# Settings 
OBJ_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
OUTPUT_DIR = "render_euler_1Ag"
IMG_WIDTH, IMG_HEIGHT = 640, 480
STEP_DEG = 5
DIST = 2.5  # Distance multiplier from object center

os.makedirs(OUTPUT_DIR, exist_ok=True)
mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
mesh.compute_vertex_normals()

# Renderer setup
render = o3d.visualization.rendering.OffscreenRenderer(IMG_WIDTH, IMG_HEIGHT)
render.scene.set_background([0, 0, 0, 1])
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
render.scene.add_geometry("model", mesh, mat)

# Bounding Box and Camera
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
radius = bbox.get_max_extent()
eye_distance = radius * DIST
cam = render.scene.camera
def euler_to_rot_matrix(yaw, pitch, roll):
    # Angles in degrees
    y, p, r = np.radians([yaw, pitch, roll])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)]
    ])
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

# Define your desired Euler angles
yaw, pitch, roll = 0, 90, 0  # degrees

# Convert to rotation matrix and compute eye position
R = euler_to_rot_matrix(yaw, pitch, roll)
forward = R @ np.array([0, 0, 1])  # looking forward
eye = center + eye_distance * forward
up = R @ np.array([0, 1, 0])       # rotated "up" vector

# Look and render
cam.look_at(center, eye, up)
img = render.render_to_image()
img_np = np.asarray(img)

# Save the image
filename = os.path.join(OUTPUT_DIR, f"euler_{yaw}_{pitch}_{roll}.png")
cv2.imwrite(filename, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
print(f"  -> saved: {filename}")