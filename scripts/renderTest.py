import open3d as o3d
import numpy as np
import cv2
import os

# === Settings ===
OBJ_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
OUTPUT_PATH = "render_test_output.png"
IMG_WIDTH, IMG_HEIGHT = 640, 480

# === Load mesh and scale ===
mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
mesh.compute_vertex_normals()
mesh.scale(0.001, center=mesh.get_center())  # convert mm to meters

# === Renderer setup ===
render = o3d.visualization.rendering.OffscreenRenderer(IMG_WIDTH, IMG_HEIGHT)
render.scene.set_background([0, 0, 0, 1])

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
render.scene.add_geometry("model", mesh, mat)

# === Compute camera view ===
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
radius = bbox.get_max_extent()

eye = center + np.array([0.0, 0.0, 0.2])  # Camera placed 1m in front
up = np.array([0, 1, 0])

render.scene.camera.look_at(center, eye, up)

# === Render and save ===
img = render.render_to_image()
img_np = np.asarray(img)
cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
print(f"✅ Render saved to: {OUTPUT_PATH}")
