import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d

# === Settings ===
IMG_MASK_PATH = "/home/simon/Documents/MVSR Lab/mvsr/output/segmentation_mask_1.png"
IMG_REAL_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/rgb/1.png"
OBJ_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
CAMERA_MATRIX = np.array([[616.741, 0, 324.818],
                          [0, 616.920, 238.046],
                          [0, 0, 1]], dtype=np.float32)

# === Helper Functions ===
def load_obj_vertices(obj_path):
    print(f"Loading 3D points from: {obj_path}")
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
    print(f"  -> Loaded {len(vertices)} points.")
    return np.array(vertices)

def fps_sample(vertices, n_points=100):
    print(f"Performing Farthest Point Sampling ({n_points} points)...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    down = pcd.voxel_down_sample(voxel_size=0.005)
    down_points = np.asarray(down.points)
    if len(down_points) < n_points:
        print(f"  -> Warning: only {len(down_points)} points after downsampling, reducing target count.")
        return down_points
    indices = np.linspace(0, len(down_points) - 1, n_points, dtype=int)
    sampled = down_points[indices]
    print(f"  -> FPS completed with {len(sampled)} points.")
    return sampled

# === Load Mask and Real Image ===
print("Loading segmentation mask and real image...")
mask_img = cv2.imread(IMG_MASK_PATH, cv2.IMREAD_GRAYSCALE)
real_img = cv2.imread(IMG_REAL_PATH, cv2.IMREAD_GRAYSCALE)

# === Find contours in mask ===
_, seg_thresh = cv2.threshold(mask_img, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(seg_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"  -> Detected {len(contours)} objects.")

# === Prepare SIFT ===
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# === Prepare 3D object keypoints ===
vertices = load_obj_vertices(OBJ_PATH)
keypoints_3d = fps_sample(vertices, 100)

# Create dummy 2D image from 3D keypoints for SIFT detection
x, y = keypoints_3d[:, 0], keypoints_3d[:, 1]
x -= x.min(); y -= y.min()
x, y = (x * 10).astype(int), (y * 10).astype(int)
render_img = np.zeros((int(y.max() + 20), int(x.max() + 20)), dtype=np.uint8)
for px, py in zip(x, y):
    cv2.circle(render_img, (px, py), 2, 255, -1)
kp_render, des_render = sift.detectAndCompute(render_img, None)

# === Search best pose across all mask objects ===
best_result = {
    "inliers": 0,
    "rvec": None,
    "tvec": None,
    "matches": None,
    "contour_idx": None,
    "kp_real": None
}

for c_idx, cnt in enumerate(contours):
    mask = np.zeros_like(mask_img)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    all_kp, all_des = sift.detectAndCompute(real_img, None)
    kp_real = []
    des_real = []

    for i, kp in enumerate(all_kp):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            kp_real.append(kp)
            des_real.append(all_des[i])

    des_real = np.array(des_real) if des_real else None
    print(f"Object {c_idx} (mask/object): {len(kp_real)} keypoints")

    # Visualize keypoints per object
    img_kp = cv2.cvtColor(real_img, cv2.COLOR_GRAY2BGR)
    kp_overlay = cv2.drawKeypoints(img_kp, kp_real, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure()
    plt.imshow(cv2.cvtColor(kp_overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT Keypoints for Object {c_idx}")
    plt.axis('off')
    plt.show()

    if des_render is None or des_real is None:
        continue

    matches = bf.knnMatch(des_real, des_render, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) >= 6:
        pts_real = np.float32([kp_real[m.queryIdx].pt for m in good])
        pts_model = np.float32([keypoints_3d[m.trainIdx] for m in good])

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_model, pts_real, CAMERA_MATRIX, None)

        if success and inliers is not None and len(inliers) > best_result["inliers"]:
            best_result.update({
                "inliers": len(inliers),
                "rvec": rvec,
                "tvec": tvec,
                "matches": good,
                "contour_idx": c_idx,
                "kp_real": kp_real
            })

# === Display result and visualize ===
if best_result["rvec"] is not None:
    print("Best pose found for object", best_result["contour_idx"],
          "with", best_result["inliers"], "inliers")
    print("rvec:\n", best_result["rvec"])
    print("tvec:\n", best_result["tvec"])

    object_points = np.array([
        keypoints_3d[m.trainIdx] for m in best_result["matches"]
    ], dtype=np.float32)

    image_points, _ = cv2.projectPoints(
        object_points, best_result["rvec"], best_result["tvec"], CAMERA_MATRIX, None)

    img_vis = cv2.cvtColor(real_img, cv2.COLOR_GRAY2BGR)
    for pt in image_points:
        pt = tuple(np.round(pt[0]).astype(int))
        cv2.circle(img_vis, pt, 4, (0, 255, 0), -1)

    label = f"Object {best_result['contour_idx']} detected"
    cv2.putText(img_vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    plt.figure()
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title("Projection of Best Pose (SIFT, Object Keypoints)")
    plt.axis('off')
    plt.show()
else:
    print("No valid pose found.")
