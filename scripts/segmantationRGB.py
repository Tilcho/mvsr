import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in BGR and convert to RGB
image_bgr = cv2.imread("/home/simon/Documents/MVSR Lab/mvsr/data/rgb/0.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Show RGB histograms
colors = ('r', 'g', 'b')
plt.figure(figsize=(10, 4))
for i, col in enumerate(colors):
    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('RGB Channel Histograms')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()

# --- Yellow segmentation in RGB ---
lower_yellow = np.array([50, 30, 20])
upper_yellow = np.array([200, 150, 130])
yellow_mask = cv2.inRange(image_rgb, lower_yellow, upper_yellow)

# --- Gray segmentation in HSV ---
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
lower_hsv_gray = np.array([0, 0, 60])
upper_hsv_gray = np.array([180, 50, 200])
gray_mask = cv2.inRange(image_hsv, lower_hsv_gray, upper_hsv_gray)

# Combine both masks
combined_mask = cv2.bitwise_or(yellow_mask, gray_mask)

# Apply mask to image
segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=combined_mask)

# Show masks and result
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.title('Yellow Mask (RGB)')
plt.imshow(yellow_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Gray Mask (HSV)')
plt.imshow(gray_mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.title('Segmented Image (Yellow + Gray)')
plt.imshow(segmented)
plt.axis('off')

plt.tight_layout()
plt.show()
