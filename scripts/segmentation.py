import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# --- Create output folder if it doesn't exist ---
output_dir = "/home/simon/Documents/MVSR Lab/mvsr/output"
os.makedirs(output_dir, exist_ok=True)

# --- Load and convert image ---
image_bgr = cv2.imread("/home/simon/Documents/MVSR Lab/mvsr/data/rgb/2.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

# --- Gray object thresholds (LAB) ---
lower_gray_lab = np.array([10, 120, 120])
upper_gray_lab = np.array([100, 135, 135])
gray_mask_lab = cv2.inRange(image_lab, lower_gray_lab, upper_gray_lab)

# --- Blue background thresholds (LAB) ---
lower_blue_lab = np.array([0, 0, 0])
upper_blue_lab = np.array([255, 255, 110])
blue_mask_lab = cv2.inRange(image_lab, lower_blue_lab, upper_blue_lab)

# --- Remove blue from gray mask ---
gray_cleaned_mask = cv2.bitwise_and(gray_mask_lab, cv2.bitwise_not(blue_mask_lab))

# --- Morphological cleanup ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
gray_cleaned_mask = cv2.morphologyEx(gray_cleaned_mask, cv2.MORPH_OPEN, kernel)

# --- Label connected components ---
labeled_mask = label(gray_cleaned_mask > 0)

# --- Filter by region size ---
min_area = 5000  # adjust as needed
max_area = 20000  # adjust as needed
filtered_mask = np.zeros_like(gray_cleaned_mask)

for region in regionprops(labeled_mask):
    if region.area >= min_area and region.area <= max_area:
        coords = region.coords
        for y, x in coords:
            filtered_mask[y, x] = 255

# --- Optional: Label final filtered mask for color visualization ---
final_labeled = label(filtered_mask > 0)
colored_labels = label2rgb(final_labeled, image_rgb, bg_label=0)

# --- Apply final mask to original image ---
segmented_filtered = cv2.bitwise_and(image_rgb, image_rgb, mask=filtered_mask)

cv2.imwrite(os.path.join(output_dir, "segmentation_mask_2.png"), filtered_mask)
cv2.imwrite(os.path.join(output_dir, "segmented_picture_2.png"), segmented_filtered)
cv2.imwrite(os.path.join(output_dir, "label_mask_2.png"), final_labeled.astype(np.uint16))  # for multi-label IDs
print("Saved segmentation_mask_2.png, segmented_picture_2.png and label_mask_2.png")


# --- Show results ---
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Filtered Gray Mask")
plt.imshow(filtered_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmented Gray Objects")
plt.imshow(segmented_filtered)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Labeled Regions Overlay")
plt.imshow(colored_labels)
plt.axis('off')

plt.tight_layout()
plt.show()
