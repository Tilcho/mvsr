import os
import cv2
import numpy as np
from math import radians
import matplotlib.pyplot as plt


# === Settings ===
IMG_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/rgb/0enhanced.png"
YOLO_PATH = "/home/simon/Documents/MVSR Lab/mvsr/detections_0.txt"
RENDER_DIR = "/home/simon/Documents/MVSR Lab/mvsr/render_euler_1Ag"

IMG_WIDTH, IMG_HEIGHT = 640, 480
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# === Camera intrinsics ===
K = np.array([
    [616.741455078125, 0.0, 324.817626953125],
    [0.0, 616.919677734375, 238.0455780029297],
    [0.0, 0.0, 1.0]
])

# === Load real image and YOLO detections ===
print("[INFO] Loading real image and detections...")
img_real = cv2.imread(IMG_PATH)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB))
plt.title("Original image no changes")
plt.axis("off")
plt.show()
img_gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY).astype(np.uint8)



detections = []
with open(YOLO_PATH, "r") as f:
    for line in f:
        class_id, conf, x1, y1, x2, y2 = map(float, line.strip().split())
        if int(class_id) == 0:
            detections.append([int(x1), int(y1), int(x2), int(y2)])

print(f"[INFO] Found {len(detections)} class 0 detections.")

results = []

# === Load rendered reference features ===
print("[INFO] Loading rendered features from:", RENDER_DIR)
rendered_features = []
for filename in os.listdir(RENDER_DIR):
    if filename.endswith(".png"):
        path = os.path.join(RENDER_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kps, desc = sift.detectAndCompute(img, None)
        if desc is not None:
            print(f"[DEBUG] {filename}: {len(kps)} keypoints")
            rendered_features.append({
                "filename": filename,
                "keypoints": kps,
                "descriptors": desc
            })
        else:
            print(f"[WARN] No descriptors found in {filename}")

        img_with_kp = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
        plt.title(f"Render {filename} with SIFT keypoints")
        plt.axis("off")
        plt.show()
        '''
print(f"\n-----------------------------------------------------------------\n\n")

# === Match and estimate pose ===
for i, (x1, y1, x2, y2) in enumerate(detections):
    print(f"[INFO] Processing detection {i + 1}: bbox=({x1},{y1},{x2},{y2})")
    #clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    #enhanced = clahe.apply(img_gray)

    crop = img_gray[y1:y2, x1:x2]
    kps_crop, desc_crop = sift.detectAndCompute(crop, None)
    if desc_crop is None:
        print("[WARN] No descriptors in cropped detection area.")
        continue
    print(f"[DEBUG] Detection {i + 1}: {len(kps_crop)} keypoints in crop")
    img_with_kp = cv2.drawKeypoints(crop, kps_crop, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box with SIFT keypoints")
    plt.axis("off")
    plt.show()

    best_match = None
    best_score = float("inf")
    best_filename = None

    for ref in rendered_features:
        matches = bf.match(desc_crop, ref["descriptors"])
        score = sum([m.distance for m in matches])
        print(f"[DEBUG] Comparing to {ref['filename']}: {len(matches)} matches, total score = {score:.2f}")

        if best_match is None or score > best_score:
            best_match = matches
            best_score = score
            best_filename = ref["filename"]

    if best_filename:
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        print(f"[INFO] Best match for detection {i + 1}: {best_filename} with score {best_score:.2f}")
        results.append({
            "bbox": [x1, y1, x2, y2],
            "center": [center_x, center_y],
            "matched_template": best_filename,
            "num_keypoints_crop": len(kps_crop),
            "num_matches": len(best_match)
        })
        # Load the best matched rendered image (grayscale)
        rendered_path = os.path.join(RENDER_DIR, best_filename)
        rendered_img = cv2.imread(rendered_path, cv2.IMREAD_GRAYSCALE)

        # Detect SIFT features in both
        kps_crop, desc_crop = sift.detectAndCompute(crop, None)
        kps_rendered, desc_rendered = sift.detectAndCompute(rendered_img, None)

        # Match
        matches = bf.match(desc_crop, desc_rendered)

        # Compute angle differences
        angle_diffs = []
        for m in matches:
            angle1 = kps_crop[m.queryIdx].angle
            angle2 = kps_rendered[m.trainIdx].angle
            diff = (angle2 - angle1 + 360) % 360  # Ensure result is [0, 360)
            angle_diffs.append(diff)

        # Estimate dominant rotation
        if angle_diffs:
            estimated_rotation = np.median(angle_diffs)
            print(f"[INFO] Estimated rotational difference: {estimated_rotation:.2f}°")
        else:
            print("[WARN] No valid matches for rotation estimation.")


print(f"[INFO] Pose estimation complete. Results {best_filename} saved.")

rendered_pose_img = cv2.imread(os.path.join(RENDER_DIR, best_filename))
kps_rend, dec_rend = sift.detectAndCompute(rendered_pose_img, None)
rend_img_with_kp = cv2.drawKeypoints(rendered_pose_img, kps_rend, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(rend_img_with_kp, cv2.COLOR_BGR2RGB))
plt.title("Calculated Pose of the Object:")
plt.axis("off")
plt.show()