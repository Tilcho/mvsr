import cv2
import numpy as np
import open3d as o3d
from skimage.metrics import structural_similarity as ssim

# Load camera intrinsics
K = np.array([
    [616.7415, 0.0, 324.8176],
    [0.0, 616.9197, 238.0456],
    [0.0, 0.0, 1.0]
])

# Load segmentation mask
mask_path = "/home/simon/Documents/MVSR Lab/mvsr/output/segmentation_mask_1.png"
print(f"[INFO] Loading mask: {mask_path}")
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print(f"[ERROR] Failed to load mask image.")
    exit()
colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"[INFO] Found {len(contours)} contours.")

# Load .obj mesh and sample 3D points
obj_path = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
print(f"[INFO] Loading mesh: {obj_path}")
mesh = o3d.io.read_triangle_mesh(obj_path)
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)

np.random.seed(42)
indices = np.random.choice(len(vertices), size=10, replace=False)
object_points = vertices[indices].astype(np.float32)
print(f"[INFO] Sampled {len(object_points)} 3D points from mesh.")

def draw_axes(img, imgpts):
    imgpts = imgpts.astype(int).reshape(-1, 2)
    origin = tuple(imgpts[0])
    cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 2)
    cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 2)
    cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 2)
    return img

best_score = -1
best_output = None

for i, cnt in enumerate(contours):
    print(f"[INFO] Processing contour {i + 1}/{len(contours)}")
    cnt = cnt.squeeze()
    if len(cnt.shape) != 2 or cnt.shape[0] < len(object_points):
        print(f"[WARN] Contour {i + 1} skipped due to insufficient points.")
        continue
    
    sampled = cnt[np.linspace(0, len(cnt) - 1, len(object_points)).astype(int)].astype(np.float32)
    
    success, rvec, tvec = cv2.solvePnP(object_points, sampled, K, None)
    if not success:
        print(f"[WARN] solvePnP failed for contour {i + 1}.")
        continue
    print(f"[INFO] solvePnP successful for contour {i + 1}.")

    axis = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])  # Enlarged axis
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, None)
    
    test_img = draw_axes(colored_mask.copy(), imgpts)
    score = ssim(mask, cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY))
    print(f"[INFO] Contour {i + 1} SSIM score: {score:.4f}")

    if score > best_score:
        print(f"[INFO] New best match found with contour {i + 1}.")
        best_score = score
        best_output = test_img.copy()
        best_contour = cnt.copy()
        best_index = i


# Save result
if best_output is not None:
    # Draw and label matched contour
    labeled = best_output.copy()
    cv2.drawContours(labeled, [best_contour.reshape(-1, 1, 2)], -1, (0, 255, 255), 2)  # Yellow
    M = cv2.moments(best_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(labeled, "1Agray", (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Red

    cv2.imwrite("pose_output.png", labeled)
    print("[SUCCESS] Saved: pose_output.png with contour label")

