import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def SIFT():
    root = os.getcwd()
    imgPath = "/home/simon/Documents/MVSR Lab/mvsr/data/rgb/0.png"
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints = sift.detect(imgGray, None)
    imgGray = cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

if __name__ == "__main__":
    SIFT()