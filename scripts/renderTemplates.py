import os
import json
import numpy as np
import cv2
import trimesh
import pyrender

# === CONFIGURATION ===
obj_path = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
output_dir = "linemod_templates"
os.makedirs(output_dir, exist_ok=True)

save_normals = True
save_metadata = True

# === LOAD & CENTER MESH ===
mesh = trimesh.load(obj_path)
mesh_centered = mesh.copy()
mesh_centered.vertices -= mesh_centered.bounding_box.centroid
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_centered, smooth=False)

# === SCENE SETUP ===
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
scene.add(pyrender_mesh)

# Camera and light nodes
camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
camera_node = scene.add(camera, pose=np.eye(4))
light_node = scene.add(light, pose=np.eye(4))

renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

# === PARAMETERS ===
num_azimuth = 10
num_elevation = 10
radius = 300
index = 0
metadata = {}

for elev_deg in np.linspace(15, 75, num_elevation):
    elev = np.deg2rad(elev_deg)
    for azim_deg in np.linspace(0, 360, num_azimuth, endpoint=False):
        azim = np.deg2rad(azim_deg)

        # Spherical coordinates
        x = radius * np.cos(elev) * np.cos(azim)
        y = radius * np.sin(elev)
        z = radius * np.cos(elev) * np.sin(azim)
        camera_position = np.array([x, y, z])

        # Compute camera look-at matrix
        target = np.array([0, 0, 0])
        forward = (target - camera_position)
        forward /= np.linalg.norm(forward)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = true_up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = camera_position

        # Update poses
        scene.set_pose(camera_node, pose=cam_pose)
        scene.set_pose(light_node, pose=cam_pose)

        # Render RGB + depth
        color, depth = renderer.render(scene)
        rgb_img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        mask_img = (depth > 0).astype(np.uint8) * 255

        # File paths
        basename = f"{index:04d}"
        rgb_path = os.path.join(output_dir, f"{basename}_rgb.png")
        mask_path = os.path.join(output_dir, f"{basename}_mask.png")
        pose_path = os.path.join(output_dir, f"{basename}_pose.npy")
        normal_path = os.path.join(output_dir, f"{basename}_normals.png")

        # Save outputs
        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(mask_path, mask_img)
        np.save(pose_path, cam_pose)

        # Save normals if needed
        if save_normals:
            normal_scene = pyrender.Scene()
            normal_scene.add(pyrender_mesh)
            normal_scene.add(camera, pose=cam_pose)
            normal_scene.add(light, pose=cam_pose)
            normals_rendered, _ = renderer.render(normal_scene, flags=pyrender.RenderFlags.FLAT)
            cv2.imwrite(normal_path, cv2.cvtColor(normals_rendered, cv2.COLOR_RGB2BGR))

        # Add to metadata
        if save_metadata:
            metadata[basename] = {
                "rgb": f"{basename}_rgb.png",
                "mask": f"{basename}_mask.png",
                "pose": f"{basename}_pose.npy",
                "normals": f"{basename}_normals.png" if save_normals else None,
                "azimuth_deg": float(azim_deg),
                "elevation_deg": float(elev_deg)
            }

        print(f"[{basename}] saved")
        index += 1

renderer.delete()

# Save metadata
if save_metadata:
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

print(f"✅ Done. Saved {index} views and metadata to: {output_dir}")
