import os
import cv2
import numpy as np
import json
from math import radians
import matplotlib.pyplot as plt


# === Settings ===
IMG_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/rgb/0enhanced.png"
YOLO_PATH = "/home/simon/Documents/MVSR Lab/mvsr/detections_0.txt"
RENDER_DIR = "/home/simon/Documents/MVSR Lab/mvsr/render_euler_1Ag"
OUTPUT_JSON = "pose_results.json"

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
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

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
'''
# === Save to JSON ===
print(f"[INFO] Writing results to {OUTPUT_JSON}")
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)
'''
print(f"[INFO] Pose estimation complete. Results {best_filename} saved.")
