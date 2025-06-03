import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


# === Camera intrinsics ===
K = np.array([
    [616.741455078125, 0.0, 324.817626953125],
    [0.0, 616.919677734375, 238.0455780029297],
    [0.0, 0.0, 1.0]
])

# === Use absolute paths ===
rgb_path = '/home/simon/Documents/MVSR Lab/mvsr/data/rgb/0.png'
depth_path = '/home/simon/Documents/MVSR Lab/mvsr/data/depth/d0_8bit.png'

# === Load images ===
rgb = cv2.imread(rgb_path)
if rgb is None:
    raise FileNotFoundError(f"Could not read RGB image at {rgb_path}")
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

depth = cv2.imread(depth_path, -1)
if depth is None:
    raise FileNotFoundError(f"Could not read depth image at {depth_path}")
depth = depth.astype(np.float32)

# === Show 2D RGB and depth images ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth, cmap='gray')
plt.title("Depth Image")
plt.axis('off')

plt.tight_layout()
plt.show()

# === Intrinsics ===
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
h, w = depth.shape
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
x3 = (xx - cx) * depth / fx
y3 = (yy - cy) * depth / fy
z3 = depth

# === 3D point cloud from depth ===
points_3d = np.dstack((x3, y3, z3))
mask = (depth > 0)
pc = points_3d[mask]
colors = rgb[mask]

# === Open3D PointCloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc / 1000.0)  # assuming depth in mm
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

# === Optional: Save to file ===
o3d.io.write_point_cloud("pointcloud.ply", pcd)
print("Saved point cloud to 'pointcloud.ply'.")

try:
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")
except:
    print("[!] OpenGL viewer failed (Wayland). Showing fallback 3D scatter plot with matplotlib...")

    # Downsample to reduce rendering cost
    sample_step = 100
    pc_sample = pc[::sample_step]
    color_sample = colors[::sample_step] / 255.0

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2],
               c=color_sample, s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Point Cloud (Matplotlib fallback)")
    plt.tight_layout()
    plt.show()
