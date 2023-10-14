# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:51:36 2023

@author: Sarwar
"""

# Another implementation
# https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/

import numpy as np
import cv2 as cv
from datetime import datetime
from matplotlib import pyplot as plt

# feature matching
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)



# path of image 1
path1 = "Image_026.jpg"

# path of image 2
path2 = "Image_029.jpg"

# img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
img1 = cv.imread(path1)
img2 = cv.imread(path2)

img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB

# compute the descriptors with ORB
kp1, desc1 = orb.detectAndCompute(img1, None)
start = datetime.now()
# find the keypoints with ORB

# compute the descriptors with ORB
kp2, desc2 = orb.detectAndCompute(img2, None)

matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

#img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[300:600], img2,  matchColor=match_color,singlePointColor=pt_color, flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[0:600], img2,  flags=2)

cv.imshow('ORB Regular Image', img3)

cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 2 general image : {td:.03f}ms")
print(len(kp1))
print(len(kp2))
print(len(matches))
print((len(kp2)/len(kp1))*100)
#print((len(kp2_90_clkw)/len(kp1))*100)
# draw only keypoints location,not size and orientation
# img3 = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
# plt.imshow(img3), plt.show()

# ####Image rotate 90 degree clockwise end ###

#kp2, desc2 = orb.compute(img2, kp2)
start = datetime.now()
img2_90_clkw = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
kp2_90_clkw, desc2_90_clkw = orb.detectAndCompute(img2_90_clkw, None)

matches = bf.match(desc1, desc2_90_clkw)
matches = sorted(matches, key=lambda x: x.distance)


#img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[300:600], img2,  matchColor=match_color,singlePointColor=pt_color, flags=2)
img3_90_clkw = cv.drawMatches(img1, kp1, img2_90_clkw, kp2_90_clkw, matches[0:600], None,  flags=2)
end = datetime.now()
cv.imshow('ORB: 90 degree clockwise Rotated Image', img3_90_clkw)
td = (end - start).total_seconds() * 10**3
print(f"TExecution time for 90 degree rotated image : {td:.03f}ms")
cv.waitKey(0)
print(len(kp1))
print(len(kp2_90_clkw))
print(len(matches))
#print((len(kp2_180_clkw)/len(kp1))*100)
# ####Image rotate 90 degree clockwise end ###

# ####Image rotate 180 degree clockwise start ###

#kp2, desc2 = orb.compute(img2, kp2)
start = datetime.now()
img2_180_clkw = cv.rotate(img2, cv.ROTATE_180)
kp2_180_clkw, desc2_180_clkw = orb.detectAndCompute(img2_180_clkw, None)

matches = bf.match(desc1, desc2_180_clkw)
matches = sorted(matches, key=lambda x: x.distance)


#img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[300:600], img2,  matchColor=match_color,singlePointColor=pt_color, flags=2)
img3_180_clkw = cv.drawMatches(img1, kp1, img2_180_clkw, kp2_180_clkw, matches[0:600], None,  flags=2)
end = datetime.now()
cv.imshow('ORB: 180 degree clockwise Rotated Image', img3_180_clkw)
td = (end - start).total_seconds() * 10**3
print(f"TExecution time for 180 degree rotated image : {td:.03f}ms")
cv.waitKey(0)
print(len(kp1))
print(len(kp2_180_clkw))
print(len(matches))

# ####Image rotate 180 degree clockwise end ###

# ####Image rotate 90 degree anticlockwise start ###

#kp2, desc2 = orb.compute(img2, kp2)
start = datetime.now()
img2_270_clkw = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
kp2_270_clkw, desc2_270_clkw = orb.detectAndCompute(img2_270_clkw, None)

matches = bf.match(desc1, desc2_270_clkw)
matches = sorted(matches, key=lambda x: x.distance)


#img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[300:600], img2,  matchColor=match_color,singlePointColor=pt_color, flags=2)
img3_270_clkw = cv.drawMatches(img1, kp1, img2_270_clkw, kp2_270_clkw, matches[0:600], None,  flags=2)
end = datetime.now()
cv.imshow('ORB: 90 degree anti-clockwise Rotated Image', img3_270_clkw)
td = (end - start).total_seconds() * 10**3

cv.waitKey(0)
print(f"Execution time for 90 degree anticlockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp2_270_clkw))
print(len(matches))
print((len(kp2_270_clkw)/len(kp1))*100)
# ####Image rotate 180 degree clockwise end ###