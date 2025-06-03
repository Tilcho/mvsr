import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load and convert image ---
image_bgr = cv2.imread("/home/simon/Documents/MVSR Lab/mvsr/data/rgb/0.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

# --- Gray object thresholds (LAB) ---
# Gray has: low color variation (A ≈ 128, B ≈ 128), mid brightness (L)
lower_gray_lab = np.array([30, 120, 120])
upper_gray_lab = np.array([220, 135, 135])
gray_mask_lab = cv2.inRange(image_lab, lower_gray_lab, upper_gray_lab)

# --- Blue background thresholds (LAB) ---
# Blue: strong in negative B channel (below 128)
lower_blue_lab = np.array([0, 0, 0])
upper_blue_lab = np.array([255, 255, 110])
blue_mask_lab = cv2.inRange(image_lab, lower_blue_lab, upper_blue_lab)

# --- Subtract blue mask from gray mask ---
gray_cleaned_mask = cv2.bitwise_and(gray_mask_lab, cv2.bitwise_not(blue_mask_lab))

# --- Apply mask to original RGB image ---
segmented_cleaned = cv2.bitwise_and(image_rgb, image_rgb, mask=gray_cleaned_mask)

# --- Display results ---
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gray Mask w/o Blue Box")
plt.imshow(gray_cleaned_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Segmented Gray Objects")
plt.imshow(segmented_cleaned)
plt.axis('off')

plt.tight_layout()
plt.show()
