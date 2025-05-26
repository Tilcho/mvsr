import open3d as o3d
import numpy as np
import os
import json
from math import radians

# === CONFIGURATION ===
OBJ_MODEL_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
OUTPUT_DIR = "output_rendered_data"
NUM_IMAGES = 50

INTRINSICS = {
    "fx": 616.7415,
    "fy": 616.9197,
    "cx": 324.8176,
    "cy": 238.0456,
    "width": 640,
    "height": 480
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def create_renderer(width, height):
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    render.scene.set_background([0, 0, 0, 1])
    render.scene.scene.enable_sun_light(False)
    render.scene.scene.set_indirect_light_intensity(15000)

    return render

def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.scale(0.5 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    mesh.translate(-mesh.get_center())
    return mesh


def render_views(mesh, intrinsics, num_views):
    renderer = create_renderer(intrinsics.width, intrinsics.height)
    renderer.scene.add_geometry("object", mesh, o3d.visualization.rendering.MaterialRecord())

    for i in range(num_views):
        angle = i * (360.0 / num_views)
        R = mesh.get_rotation_matrix_from_xyz((radians(45), radians(angle), 0))
        T = [0, 0, 1.0]

        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsics
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T
        cam.extrinsic = extrinsic

        renderer.setup_camera(intrinsics, cam.extrinsic)

        # Render and save
        rgb = renderer.render_to_image()
        depth = renderer.render_to_depth_image()

        o3d.io.write_image(os.path.join(OUTPUT_DIR, f"rgb_{i:03d}.png"), rgb)
        o3d.io.write_image(os.path.join(OUTPUT_DIR, f"depth_{i:03d}.png"), depth)

        # Save pose
        pose = {
            "rotation_matrix": R.tolist(),
            "translation": T
        }
        with open(os.path.join(OUTPUT_DIR, f"pose_{i:03d}.json"), "w") as f:
            json.dump(pose, f)

    print(f"Saved {num_views} rendered views to {OUTPUT_DIR}")

# Run rendering
intr = get_camera_intrinsics()
mesh = load_mesh(OBJ_MODEL_PATH)
render_views(mesh, intr, NUM_IMAGES)
