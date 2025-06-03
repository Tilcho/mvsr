import cv2

def match_orb_features(rgb, template):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(rgb, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches
