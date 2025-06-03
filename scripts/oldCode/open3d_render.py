import open3d as o3d
import numpy as np
import os
import json
from math import radians

# === CONFIGURATION ===
# Path to the 3D model (OBJ file)
OBJ_MODEL_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
# Output directory for rendered images and poses
OUTPUT_DIR = "output_rendered_data"
# Number of rendered views/images to generate
NUM_IMAGES = 50

# Camera intrinsic parameters
INTRINSICS = {
    "fx": 616.7415,
    "fy": 616.9197,
    "cx": 324.8176,
    "cy": 238.0456,
    "width": 640,
    "height": 480
}

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to generate Open3D camera intrinsics object
def get_camera_intrinsics():
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(
        width=INTRINSICS["width"],
        height=INTRINSICS["height"],
        fx=INTRINSICS["fx"],
        fy=INTRINSICS["fy"],
        cx=INTRINSICS["cx"],
        cy=INTRINSICS["cy"]
    )
    return intr

# Function to create and configure an offscreen renderer
def create_renderer(width, height):
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    render.scene.set_background([0, 0, 0, 1])  # Black background
    render.scene.scene.enable_sun_light(False)  # Disable sun light
    render.scene.scene.set_indirect_light_intensity(15000)  # Set high indirect light intensity

    return render

# Function to load and preprocess the mesh model
def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()  # Compute vertex normals for rendering
    # Normalize scale of the mesh
    mesh.scale(0.5 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    # Center the mesh at the origin
    mesh.translate(-mesh.get_center())
    return mesh

# Function to render multiple views of the mesh and save images and poses
def render_views(mesh, intrinsics, num_views):
    # Create the renderer
    renderer = create_renderer(intrinsics.width, intrinsics.height)
    # Add the mesh to the rendering scene
    renderer.scene.add_geometry("object", mesh, o3d.visualization.rendering.MaterialRecord())

    # Loop over the number of views to render
    for i in range(num_views):
        # Compute rotation angle around the Y-axis
        angle = i * (360.0 / num_views)
        # Rotation: fixed tilt of 45 degrees in X, rotating in Y
        R = mesh.get_rotation_matrix_from_xyz((radians(45), radians(angle), 0))
        # Translation: move camera 1 unit along Z-axis
        T = [0, 0, 1.0]

        # Set up camera parameters
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsics
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T
        cam.extrinsic = extrinsic

        # Apply camera parameters to the renderer
        renderer.setup_camera(intrinsics, cam.extrinsic)

        # Render RGB and depth images
        rgb = renderer.render_to_image()
        depth = renderer.render_to_depth_image()

        # Save rendered images
        o3d.io.write_image(os.path.join(OUTPUT_DIR, f"rgb_{i:03d}.png"), rgb)
        o3d.io.write_image(os.path.join(OUTPUT_DIR, f"depth_{i:03d}.png"), depth)

        # Save camera pose (rotation and translation)
        pose = {
            "rotation_matrix": R.tolist(),
            "translation": T
        }
        with open(os.path.join(OUTPUT_DIR, f"pose_{i:03d}.json"), "w") as f:
            json.dump(pose, f)

    # Notify user rendering is complete
    print(f"Saved {num_views} rendered views to {OUTPUT_DIR}")

# Run rendering process
intr = get_camera_intrinsics()  # Get camera intrinsics
mesh = load_mesh(OBJ_MODEL_PATH)  # Load and process the 3D mesh
render_views(mesh, intr, NUM_IMAGES)  # Render and save the views
