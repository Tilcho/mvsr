import trimesh
import pyrender
import matplotlib.pyplot as plt
import numpy as np
import cv2
'''
STEP ONE: PREPARE RENDER VIEWS
'''
# Load your mesh
mesh = trimesh.load('/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj')
scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
scene.add(mesh_pyrender)

# Camera intrinsics matching your real camera
camera = pyrender.IntrinsicsCamera(
    fx=616.7415, fy=616.9197,
    cx=324.8176, cy=238.0456
)

# Center the object in view using its bounding box
mesh_center = mesh.bounding_box.centroid
cam_pose = np.eye(4)
# Looking along -X, with Z pointing up (OpenGL-style)
cam_pose[:3, :3] = np.array([
    [ 0,  0, -1],   # X_cam: right → -Z
    [ 0,  1,  0],   # Y_cam: up stays Y
    [ 1,  0,  0]    # Z_cam: forward → +X, so camera looks along -Z_cam → -X world
])
# Set camera position: 300 units above center
cam_pose[:3, 3] = mesh_center + np.array([-300, 0, 0])
# Use a Perspective Camera (easier to tune field of view)
camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))

# Reset scene and add updated components
scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
scene.add(camera, pose=cam_pose)

# Add light at the same position as camera
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=cam_pose)

# Render
r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
color, depth = r.render(scene)

cv2.imwrite("synthetic_render.png", color)

'''
STEP TWO: ORB KEYSPOINTS
'''
# Load label mask and grayscale synthetic image
label_mask = cv2.imread("/home/simon/Documents/MVSR Lab/mvsr/output/label_mask_2.png", cv2.IMREAD_UNCHANGED)
img_mask = cv2.imread('/home/simon/Documents/MVSR Lab//mvsr/output/segmentation_mask_2edit.png', cv2.IMREAD_GRAYSCALE)
img_synth = cv2.imread('/home/simon/Documents/MVSR Lab/mvsr/synthetic_render.png', cv2.IMREAD_GRAYSCALE)
img_real = cv2.imread('/home/simon/Documents/MVSR Lab/mvsr/data/rgb/2.png')


# Threshold mask to binary (just in case)
_, binary_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
# Convert binary mask to 3 channels
mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])
# Apply the mask to the RGB image
masked_rgb = cv2.bitwise_and(img_real, mask_3ch)

orb = cv2.ORB_create(nfeatures=1000)

kp_real, des_real = orb.detectAndCompute(masked_rgb, None)

'''
# Get label IDs (excluding background)
label_ids = np.unique(label_mask)
label_ids = label_ids[label_ids > 0]
features_by_label = {}
for label_id in label_ids:
    # Create binary mask for the region
    region_mask = (label_mask == label_id).astype(np.uint8) * 255
    # Get bounding box
    x, y, w, h = cv2.boundingRect(region_mask)
    # Crop grayscale image and mask
    cropped_region = img_synth[y:y+h, x:x+w]
    cropped_mask = region_mask[y:y+h, x:x+w]
    # Detect ORB features using the mask
    kp_real, des_real = orb.detectAndCompute(cropped_region, mask=cropped_mask)
    # Optional: Adjust keypoint coordinates back to full image space
    if kp_real is not None:
        for kp in kp_real:
            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
    features_by_label[label_id] = {
        'keypoints': kp_real,
        'descriptors': des_real
    }
    print(f"Label {label_id}: {len(kp_real) if kp_real else 0} keypoints")
'''
# Compute keypoints for synthetic image once
kp_synth, des_synth = orb.detectAndCompute(img_synth, None)

'''
STEP THREE: MATCHING
'''
#for label_id in features_by_label:

# Match descriptors using Brute-Force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_synth, des_real)

# Sort matches by distance (best first)
matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(
    img_synth, kp_synth,
    img_real, kp_real,
    matches[:20], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.imshow(img_matches, cmap='gray')
plt.title("ORB Matches")
plt.axis("off")
plt.show()