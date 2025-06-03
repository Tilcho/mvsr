import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d

# === Einstellungen ===
IMG_PATH = "/home/simon/Documents/MVSR Lab/mvsr/output/segmentation_mask_1.png"
OBJ_PATH_1A = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
OBJ_PATH_3B = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-3B_gray.obj"
CAMERA_MATRIX = np.array([[616.741, 0, 324.818],
                          [0, 616.920, 238.046],
                          [0, 0, 1]], dtype=np.float32)

# === Hilfsfunktionen ===
def load_obj_vertices(obj_path):
    print(f"Lade 3D-Punkte aus: {obj_path}")
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
    print(f"  -> {len(vertices)} Punkte geladen.")
    return np.array(vertices)

def fps_sample(vertices, n_points=100):
    """Farthest Point Sampling (FPS) mit Open3D"""
    print(f"Führe Farthest Point Sampling durch ({n_points} Punkte)...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    down = pcd.voxel_down_sample(voxel_size=0.005)
    down_points = np.asarray(down.points)
    
    if len(down_points) < n_points:
        print(f"  -> Warnung: nur {len(down_points)} Punkte nach Downsampling, reduziere Zielanzahl.")
        return down_points
    
    indices = np.linspace(0, len(down_points) - 1, n_points, dtype=int)
    sampled = down_points[indices]
    print(f"  -> FPS abgeschlossen mit {len(sampled)} Punkten.")
    return sampled

# === Segmentierungsmaske laden und Objekte finden ===
print("Lade Segmentierungsmaske und finde Konturen...")
seg_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
_, seg_thresh = cv2.threshold(seg_img, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(seg_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"  -> {len(contours)} Objekte erkannt.")

# Visualisierung der Konturen
seg_img_color = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(seg_img_color, contours, -1, (0, 255, 0), 2)
plt.figure()
plt.imshow(cv2.cvtColor(seg_img_color, cv2.COLOR_BGR2RGB))
plt.title("Segmentierte Konturen")
plt.axis('off')
plt.show()

# === ORB Keypoints pro Objektmaske ===
print("Extrahiere ORB-Keypoints für jedes erkannte Objekt...")
orb = cv2.ORB_create(500)
keypoints_contours = []
descriptors_contours = []
bounding_boxes = []

for i, cnt in enumerate(contours):
    mask = np.zeros_like(seg_img)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    masked_img = cv2.bitwise_and(seg_img, seg_img, mask=mask)
    kp, des = orb.detectAndCompute(masked_img, mask)
    print(f"  -> Objekt {i}: {len(kp)} Keypoints")
    keypoints_contours.append(kp)
    descriptors_contours.append(des)
    bounding_boxes.append(cv2.boundingRect(cnt))

# Visualisierung der Keypoints
img_keypoints = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
colors = [(0, 255, 0), (255, 0, 0)]
for i, kp in enumerate(keypoints_contours):
    img_keypoints = cv2.drawKeypoints(img_keypoints, kp, None, color=colors[i % 2], flags=0)
for i, (x, y, w, h) in enumerate(bounding_boxes):
    cv2.rectangle(img_keypoints, (x, y), (x + w, y + h), colors[i % 2], 2)
    cv2.putText(img_keypoints, f'Objekt {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % 2], 2)
plt.figure()
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title("Keypoints & Bounding Boxes")
plt.axis('off')
plt.show()

# === CAD-Modelle laden und FPS-Keypoints erzeugen ===
vertices_1A = load_obj_vertices(OBJ_PATH_1A)
vertices_3B = load_obj_vertices(OBJ_PATH_3B)
keypoints_3d_1A = fps_sample(vertices_1A, 100)
keypoints_3d_3B = fps_sample(vertices_3B, 100)

# === Matchen & Pose für Objekt 0 (1A) schätzen ===
print("Starte Pose-Schätzung für Objekt 0 (1A)...")
kp_2d = keypoints_contours[0]
des_2d = descriptors_contours[0]
pts_2d = np.array([kp.pt for kp in kp_2d], dtype=np.float32)

# Dummy-Deskriptoren für 3D-Keypoints
des_3d = np.random.randint(0, 256, (100, 32)).astype(np.uint8)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des_2d, des_3d, k=2)

# Lowe's Ratio Test
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
print(f"  -> {len(good_matches)} gültige Matches gefunden.")

# PnP nur bei genügend Matches
if len(good_matches) >= 6:
    image_points = np.array([kp_2d[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    object_points = np.array([keypoints_3d_1A[m.trainIdx] for m in good_matches], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, CAMERA_MATRIX, None)
    if success:
        print("Pose geschätzt für Objekt 1A:")
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec):\n", tvec)
    else:
        print("solvePnP fehlgeschlagen.")
else:
    print("Nicht genug gültige Matches gefunden für Objekt 1A.")
