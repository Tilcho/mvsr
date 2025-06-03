import cv2
import numpy as np
import trimesh
from skimage.measure import regionprops, label
from skimage.io import imread
from pathlib import Path

# --- Paths ---
data_dir = Path("/home/simon/Documents/MVSR Lab/mvsr/data")
model_dir = data_dir / "models"
rgb_image_path = data_dir / "rgb/1.png"
label_mask_path = Path("/home/simon/Documents/MVSR Lab/mvsr/output/label_mask_1.png")
output_path = Path("/home/simon/Documents/MVSR Lab/mvsr/output/pose_overlay.png")

# --- Camera Intrinsics (from Camera Intrinsics.txt) ---
K = np.array([
    [616.741455078125, 0.0, 324.817626953125],
    [0.0, 616.919677734375, 238.0455780029297],
    [0.0, 0.0, 1.0]
])

# --- Load label mask and image ---
label_mask = imread(label_mask_path)
rgb_image = cv2.imread(str(rgb_image_path))
draw_image = rgb_image.copy()

# --- Load models ---
model_paths = {
    "1Agray": model_dir / "morobot-s_Achse-1A_gray.obj",
    "3Bgray": model_dir / "morobot-s_Achse-3B_gray.obj"
}
models = {name: trimesh.load_mesh(str(path)) for name, path in model_paths.items()}

# --- Function: sample N well-distributed 3D vertices from mesh ---
def sample_3d_keypoints(mesh, n_points=12):
    return mesh.sample(n_points)

# --- Function: extract 2D contour points ---
def extract_2d_keypoints(mask, label_id, n_points=12):
    region_mask = (mask == label_id).astype(np.uint8)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        points = np.squeeze(approx)
        if points.ndim == 1:
            points = points[np.newaxis, :]
        if len(points) > n_points:
            idx = np.linspace(0, len(points)-1, n_points, dtype=int)
            return points[idx].astype(np.float32)
        return points.astype(np.float32)
    return None

# --- Iterate over labeled regions ---
object_results = []
label_ids = np.unique(label_mask)
label_ids = label_ids[label_ids > 0]  # exclude background

# --- Coordinate axes in object frame (for drawing) ---
axis_3D = np.float32([
    [0, 0, 0],    # origin
    [40, 0, 0],   # x-axis
    [0, 40, 0],   # y-axis
    [0, 0, 40]    # z-axis
])

matched_models = set()

for label_id in label_ids:
    print(f"Processing region: {label_id}")
    keypoints_2D = extract_2d_keypoints(label_mask, label_id, n_points=12)
    if keypoints_2D is None or len(keypoints_2D) < 4:
        print(" - Skipped: insufficient 2D points")
        continue

    remaining_models = [m for m in models if m not in matched_models]

    best_model = None
    best_error = float('inf')
    best_pose = None

    for model_name in remaining_models:
        model = models[model_name]
        keypoints_3D = sample_3d_keypoints(model, n_points=len(keypoints_2D)).astype(np.float32)

        if len(keypoints_3D) != len(keypoints_2D):
            continue

        success, rvec, tvec = cv2.solvePnP(keypoints_3D, keypoints_2D, K, None, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            projected, _ = cv2.projectPoints(keypoints_3D, rvec, tvec, K, None)
            projected = np.squeeze(projected)
            error = np.mean(np.linalg.norm(projected - keypoints_2D, axis=1))

            if error < best_error:
                best_error = error
                best_model = model_name
                best_pose = (rvec, tvec)

    if best_model:
        print(f" - Best match: {best_model}, error={best_error:.2f}")
        matched_models.add(best_model)
        object_results.append({
            "label_id": label_id,
            "model": best_model,
            "pose": best_pose,
            "error": best_error
        })

        # Draw coordinate axes
        rvec, tvec = best_pose
        axis_imgpts, _ = cv2.projectPoints(axis_3D, rvec, tvec, K, None)
        axis_imgpts = axis_imgpts.astype(int).reshape(-1, 2)
        origin = tuple(axis_imgpts[0])
        cv2.line(draw_image, origin, tuple(axis_imgpts[1]), (0, 0, 255), 3)  # X - red
        cv2.line(draw_image, origin, tuple(axis_imgpts[2]), (0, 255, 0), 3)  # Y - green
        cv2.line(draw_image, origin, tuple(axis_imgpts[3]), (255, 0, 0), 3)  # Z - blue

        # Label the object
        cv2.putText(draw_image, best_model, (origin[0] + 10, origin[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    else:
        print(" - No matching model found")

# --- Save overlay image ---
cv2.imwrite(str(output_path), draw_image)
print(f"Pose overlay saved to: {output_path}")

# --- Print final results ---
for result in object_results:
    print(f"Object {result['label_id']}: matched {result['model']} with reproj error {result['error']:.2f}")
