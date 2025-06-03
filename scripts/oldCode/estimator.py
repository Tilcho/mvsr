import numpy as np
import cv2

def get_3d_point(x, y, depth, K):
    z = depth[y, x] / 1000.0
    if z == 0:
        return None
    X = (x - K[0, 2]) * z / K[0, 0]
    Y = (y - K[1, 2]) * z / K[1, 1]
    return np.array([X, Y, z], dtype=np.float32)

def estimate_pose(kp1, matches, depth, K):
    object_points = []
    image_points = []

    for m in matches[:30]:
        x, y = map(int, kp1[m.queryIdx].pt)
        pt3d = get_3d_point(x, y, depth, K)
        if pt3d is not None:
            object_points.append(pt3d)
            image_points.append(kp1[m.queryIdx].pt)

    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    if len(object_points) >= 6:
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None)
        return rvec, tvec
    return None, None
