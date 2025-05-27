import open3d as o3d
import numpy as np
import cv2
import os
from math import radians

# Settings
OBJ_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
OUTPUT_DIR = "render_euler_1Ag"
IMG_WIDTH, IMG_HEIGHT = 640, 480
DIST = 2.5

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

# Function: Euler to rotation matrix
def euler_to_rot_matrix(yaw, pitch, roll):
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

# Render every 5° in yaw and pitch
print("Rendering yaw/pitch grid...")
for pitch in range(0, 361, 18):
    for yaw in range(0, 360, 18):
        roll = 0  # keep fixed
        if yaw >= 70 and yaw <= 110: continue
        if yaw >= 250 and yaw <= 290: continue
        R = euler_to_rot_matrix(yaw, pitch, roll)
        forward = R @ np.array([0, 0, 1])  # camera looks forward in local frame
        up = R @ np.array([0, 1, 0])       # camera's up vector
        eye = center + eye_distance * forward

        cam.look_at(center, eye, up)
        img = render.render_to_image()
        img_np = np.asarray(img)

        filename = os.path.join(OUTPUT_DIR, f"yaw_{yaw}_pitch_{pitch}_roll_{roll}.png")
        cv2.imwrite(filename, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        print(f"  -> saved: {filename}")
'''
sift = cv2.SIFT_create()
rendered_features = []

for filename in os.listdir("/home/simon/Documents/MVSR Lab/mvsr/render_euler_1Ag"):
    if not filename.endswith(".png"):
        continue
    img_path = os.path.join(OUTPUT_DIR, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    rendered_features.append({
        "filename": filename,
        "keypoints": keypoints,
        "descriptors": descriptors
    })'''