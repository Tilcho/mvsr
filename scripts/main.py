from intrinsics import load_intrinsics
from loader import load_rgb_depth
from detector import match_orb_features
from estimator import estimate_pose
import cv2

K = load_intrinsics()
rgb, depth = load_rgb_depth("data/rgb/0.png", "data/depth/d0.png")
template = cv2.imread("data/templates/template0.png")

kp1, kp2, matches = match_orb_features(rgb, template)
rvec, tvec = estimate_pose(kp1, matches, depth, K)

if rvec is not None:
    cv2.drawFrameAxes(rgb, K, None, rvec, tvec, 0.05)
    cv2.imshow("Pose", rgb)
    cv2.waitKey(0)
else:
    print("Pose estimation failed.")
