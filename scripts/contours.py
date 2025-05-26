import numpy as np
import cv2
import open3d as o3d
from scipy.optimize import minimize
import os

# === File paths ===
rgb_image_path = "/home/simon/Documents/MVSR Lab/mvsr/data/rgb/1.png"
mask_image_path = "/home/simon/Documents/MVSR Lab/mvsr/output/segmentation_mask_1.png"
cad_model_paths = [
    "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj",
    "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-3B_gray.obj"
]

# === Image size ===
IMG_WIDTH, IMG_HEIGHT = 640, 480

# === Load input images ===
rgb_image = cv2.imread(rgb_image_path)
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
image_h, image_w = mask_image.shape

# === Find contours from segmentation mask ===
contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Load and scale meshes ===
meshes = [o3d.io.read_triangle_mesh(path) for path in cad_model_paths]
for mesh in meshes:
    mesh.compute_vertex_normals()
    mesh.scale(0.001, center=mesh.get_center())  # mm → meters

# === Look-at style renderer ===
def render_projection(mesh, R, t, image_size):
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    mesh_copy.transform(T)

    vis = o3d.visualization.rendering.OffscreenRenderer(*image_size)
    vis.scene.set_background([0, 0, 0, 0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    vis.scene.add_geometry("mesh", mesh_copy, mat)

    # Camera based on mesh bbox
    bbox = mesh_copy.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    eye = center + np.array([0.0, 0.0, 0.25])  # camera 25cm in front
    up = np.array([0, 1, 0])
    vis.scene.camera.look_at(center, eye, up)

    img = vis.render_to_image()
    return np.asarray(img)

# === Objective function for pose optimization ===
def pose_error(pose, mesh, mask, size):
    R = o3d.geometry.get_rotation_matrix_from_xyz(pose[:3])
    t = np.array(pose[3:])
    rendered = render_projection(mesh, R, t, size)
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGRA2GRAY)
    binary = (gray > 0).astype(np.uint8) * 255
    return np.sum(np.abs(mask.astype(np.float32) - binary.astype(np.float32)))

# === Overlay projection contour on RGB ===
def overlay_projection_on_image(rgb, projection):
    gray = cv2.cvtColor(projection, cv2.COLOR_BGRA2GRAY)
    binary = (gray > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = rgb.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay

# === Run pose optimization and overlay ===
initial_pose = np.array([0, 0, 0, 0, 0, 0.1])  # starting 10cm forward
results = []

for i, mesh in enumerate(meshes):
    if i >= len(contours):
        break

    mask_single = np.zeros_like(mask_image)
    cv2.drawContours(mask_single, [contours[i]], -1, 255, -1)

    res = minimize(
        pose_error,
        initial_pose,
        args=(mesh, mask_single, (IMG_WIDTH, IMG_HEIGHT)),
        method='Powell',
        options={'maxiter': 100}
    )
    results.append(res)

    # Final rendering
    R = o3d.geometry.get_rotation_matrix_from_xyz(res.x[:3])
    t = np.array(res.x[3:])
    proj = render_projection(mesh, R, t, (IMG_WIDTH, IMG_HEIGHT))

    overlay = overlay_projection_on_image(rgb_image, proj)
    output_path = f"overlay_object_{i+1}.png"
    cv2.imwrite(output_path, overlay)

    print(f"[Object {i+1}] Pose: {res.x}")
    print(f"[Object {i+1}] Overlay saved to: {output_path}")

print("✅ Pose estimation complete.")
