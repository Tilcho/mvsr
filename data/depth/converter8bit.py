#!/usr/bin/env python3

import cv2
import numpy as np

# Loop over d0 to d9
for i in range(10):
    input_filename = f'd{i}.png'
    output_filename = f'd{i}_8bit.png'

    # Load the 16-bit image (preserve depth)
    img = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Could not load {input_filename}")
        continue

    # Normalize the image to 8-bit range
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    img_8bit = np.uint8(img_8bit)

    # Save the result
    cv2.imwrite(output_filename, img_8bit)
    print(f"Saved {output_filename}")
