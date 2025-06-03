import os
import cv2
import numpy as np
import json
from math import radians
import matplotlib.pyplot as plt

IMG_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/rgb/0enhanced.png"
YOLO_PATH = "/home/simon/Documents/MVSR Lab/mvsr/detections_0.txt"

IMG_WIDTH, IMG_HEIGHT = 640, 480
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

img_real = cv2.imread(IMG_PATH)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB))
plt.title("Original image no changes")
plt.axis("off")
plt.show()

img_gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY).astype(np.uint8)
kps_real, desc_real = sift.detectAndCompute(img_gray, None)
img_with_kp = cv2.drawKeypoints(img_gray, kps_real, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
plt.title("Original image no changes")
plt.axis("off")
plt.show()


# Crop detection region
x1, y1, x2, y2 = 411, 125, 537, 263
crop = img_gray[y1:y2, x1:x2]
kps_crop, desc_crop = sift.detectAndCompute(crop, None)
if desc_crop is None:
    print("[WARN] No descriptors in cropped detection area.")

# Visualize keypoints
img_with_kp = cv2.drawKeypoints(crop, kps_crop, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
plt.title("Bounding Box with SIFT keypoints")
plt.axis("off")
plt.show()