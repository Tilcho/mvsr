import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d

# === Einstellungen ===
IMG_PATH = "/home/simon/Documents/MVSR Lab/mvsr/output/segmentation_mask_1.png"
FEATURE_FILE = "render_features.npz"
CAMERA_MATRIX = np.array([[616.741, 0, 324.818],
                          [0, 616.920, 238.046],
                          [0, 0, 1]], dtype=np.float32)

# === Segmentierungsmaske laden und Konturen finden ===
print("Lade Segmentierungsmaske und finde Konturen...")
seg_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
_, seg_thresh = cv2.threshold(seg_img, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(seg_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"  -> {len(contours)} Objekte erkannt.")

# === Feature-Datei laden ===
data = np.load(FEATURE_FILE, allow_pickle=True)
des_rendered_list = data['descriptors']

# === ORB vorbereiten ===
orb = cv2.ORB_create(500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# === Beste Pose suchen über alle Masken-Objekte ===
best_result = {
    "inliers": 0,
    "rvec": None,
    "tvec": None,
    "matches": None,
    "contour_idx": None,
    "view_idx": None,
    "kp_real": None
}

for c_idx, cnt in enumerate(contours):
    mask = np.zeros_like(seg_img)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    masked_img = cv2.bitwise_and(seg_img, seg_img, mask=mask)
    kp_real, des_real = orb.detectAndCompute(masked_img, mask)
    print(f"Objekt {c_idx}: {len(kp_real)} Keypoints")

    view_idx = 4
    des_render = des_rendered_list[view_idx]
    if des_render is None or des_real is None:
        continue

    matches = bf.knnMatch(des_real, des_render, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]  # match descriptors  # fixed indentation

    if len(good) >= 6:
        pts_real = np.float32([kp_real[m.queryIdx].pt for m in good])
        pts_model = np.float32([[kp_real[m.queryIdx].pt[0], kp_real[m.queryIdx].pt[1], 0] for m in good])

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_model, pts_real, CAMERA_MATRIX, None)

        if success and inliers is not None and len(inliers) > best_result["inliers"]:
            best_result.update({
                "inliers": len(inliers),
                "rvec": rvec,
                "tvec": tvec,
                "matches": good,
                "contour_idx": c_idx,
                "view_idx": view_idx,
                "kp_real": kp_real
            })

# === Ergebnis anzeigen und visualisieren ===
if best_result["rvec"] is not None:
    print("Beste Pose gefunden für Objekt", best_result["contour_idx"], "aus View", best_result["view_idx"],
          "mit", best_result["inliers"], "Inliers")
    print("rvec:\n", best_result["rvec"])
    print("tvec:\n", best_result["tvec"])

    object_points = np.array([
        [best_result["kp_real"][m.queryIdx].pt[0], best_result["kp_real"][m.queryIdx].pt[1], 0]
        for m in best_result["matches"]
    ], dtype=np.float32)

    image_points, _ = cv2.projectPoints(
        object_points, best_result["rvec"], best_result["tvec"], CAMERA_MATRIX, None)

    img_vis = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
    for pt in image_points:
        pt = tuple(np.round(pt[0]).astype(int))
        cv2.circle(img_vis, pt, 4, (0, 255, 0), -1)

    label = f"Objekt {best_result['contour_idx']} erkannt (View {best_result['view_idx']})"
    cv2.putText(img_vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    plt.figure()
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title("Projektion der besten Pose")
    plt.axis('off')
    plt.show()
else:
    print("Keine valide Pose gefunden.")
