import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
from PIL import Image
import os

# Paths
model_path = "data/models/morobot-s_Achse-1A_gray.obj"
output_path = "data/templates/template0.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load mesh
mesh = o3d.io.read_triangle_mesh(model_path)
if not mesh.has_triangles():
    raise RuntimeError("Mesh is empty or failed to load.")

# Create scene and renderer
scene = rendering.Open3DScene(rendering.OffscreenRenderer(512, 512).scene)
scene.add_geometry("object", mesh, rendering.MaterialRecord())

# Create renderer
renderer = rendering.OffscreenRenderer(512, 512)
renderer.scene.add_geometry("object", mesh, rendering.MaterialRecord())

# Setup camera
bounds = mesh.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = bounds.get_extent().max()
cam_pos = center + [0, 0, extent * 2]
up = [0, -1, 0]

renderer.setup_camera(60.0, bounds, center)
renderer.scene.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background

# Render image
img_o3d = renderer.render_to_image()
img = np.asarray(img_o3d)

# Save image
Image.fromarray(img).save(output_path)
print(f"Template saved to: {output_path}")
