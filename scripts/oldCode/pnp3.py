import cv2
import numpy as np
import open3d as o3d

# Camera intrinsics
K = np.array([
    [616.7415, 0.0, 324.8176],
    [0.0, 616.9197, 238.0456],
    [0.0, 0.0, 1.0]
])

# Load segmentation mask
mask_path = "/home/simon/Documents/MVSR Lab/mvsr/output/segmentation_mask_1.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("[ERROR] Failed to load mask.")
    exit()
colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Load and sort contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
print(f"[INFO] Found {len(contours)} contours.")

# Load models
models = {
    "Object 1A": "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj",
    "Object 3B": "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-3B_gray.obj"
}

model_vertices = {}
for label, path in models.items():
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    model_vertices[label] = vertices.astype(np.float32)
    print(f"[INFO] Loaded {label} with {len(vertices)} vertices")

def draw_axes(img, imgpts):
    imgpts = imgpts.astype(int).reshape(-1, 2)
    origin = tuple(imgpts[0])
    cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 2)
    cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 2)
    cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 2)
    return img

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

output = colored_mask.copy()

for i, cnt in enumerate(contours):
    cnt = cnt.squeeze()
    if len(cnt.shape) != 2 or len(cnt) < 6:
        print(f"[WARN] Skipping contour {i + 1}, not enough points.")
        continue

    # Uniformly sample contour points
    sampled_2d = cnt[np.linspace(0, len(cnt) - 1, 30).astype(int)].astype(np.float32)

    best_iou = -1
    best_label = ""
    best_rvec, best_tvec = None, None
    best_contour = cnt
    best_proj_poly = None

    for label, obj_pts in model_vertices.items():
        # Uniformly sample same number of 3D points
        indices = np.linspace(0, len(obj_pts) - 1, len(sampled_2d)).astype(int)
        pnp_3d = obj_pts[indices]

        # Solve pose
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pnp_3d,
            imagePoints=sampled_2d,
            cameraMatrix=K,
            distCoeffs=None,
            reprojectionError=8.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            continue

        # Project full mesh
        proj_2d, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
        proj_2d = proj_2d.squeeze().astype(np.int32)

        # Build masks
        proj_mask = np.zeros_like(mask, dtype=np.uint8)
        contour_mask = np.zeros_like(mask, dtype=np.uint8)
        if len(proj_2d.shape) != 2 or proj_2d.shape[0] < 3:
            continue
        cv2.fillConvexPoly(proj_mask, cv2.convexHull(proj_2d), 255)
        cv2.drawContours(contour_mask, [cnt.reshape(-1, 1, 2)], -1, 255, -1)

        iou = compute_iou(proj_mask > 0, contour_mask > 0)
        print(f"[INFO] Contour {i + 1} vs {label}: IoU = {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            best_label = label
            best_rvec, best_tvec = rvec, tvec
            best_proj_poly = cv2.convexHull(proj_2d)

    if best_label:
        print(f"[MATCH] Contour {i + 1} matched to {best_label} (IoU={best_iou:.4f})")
        # Draw projected object shape
        cv2.polylines(output, [best_proj_poly], isClosed=True, color=(255, 255, 0), thickness=2)

        # Draw and label contour
        cv2.drawContours(output, [best_contour.reshape(-1, 1, 2)], -1, (0, 255, 255), 2)
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, best_label, (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw axes
        axis = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        imgpts, _ = cv2.projectPoints(axis, best_rvec, best_tvec, K, None)
        output = draw_axes(output, imgpts)

# Save result
cv2.imwrite("pose_output.png", output)
print("[SUCCESS] Saved: pose_output.png")
