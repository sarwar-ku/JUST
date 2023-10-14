# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:38:54 2023

@author: Sarwar
"""

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

# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp1 = star.detect(img1, None)
# compute the descriptors with BRIEF
kp1, desc1 = brief.compute(img1, kp1)
start = datetime.now()
# find the keypoints with STAR
kp2 = star.detect(img2, None)
# compute the descriptors with BRIEF
kp2, desc2 = brief.compute(img2, kp2)

matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[300:600], img2, flags=2)

cv.imshow('BRIEF: Regular Image', img3)

cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 2 general image : {td:.03f}ms")
print(len(kp1))
print(len(kp2))
print(len(matches))
#print((len(kp2)/len(kp1))*100)
# draw only keypoints location,not size and orientation

# ####Image rotate 90 degree clockwise start ###
start = datetime.now()
img_90_clkw = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
'''cv.imshow('BRIEF', img_90_clkw)
cv.waitKey(0)'''

# find the keypoints with STAR for clockwise 90 degree rotated image
kp_90_clkw = star.detect(img_90_clkw, None)
# compute the descriptors with BRIEF
kp_90_clkw, desc_90_clkw = brief.compute(img_90_clkw, kp_90_clkw)

matches = bf.match(desc1, desc_90_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_90_clkw = cv.drawMatches(img1, kp1, img_90_clkw, kp_90_clkw, matches[300:600], img2, flags=2)
cv.imshow('BRIEF: 90 degree clockwise Rotated Image', img3_90_clkw)
cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"TExecution time for 90 degree rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp_90_clkw))
print(len(matches))
#print((len(kp2_90_clkw)/len(kp1))*100)

# ####Image rotate 90 degree clockwise end ###

# ####Image rotate 180 degree clockwise start ###
start = datetime.now()
img_180_clkw = cv.rotate(img2, cv.ROTATE_180)
'''cv.imshow('BRIEF', img_90_clkw)
cv.waitKey(0)'''

# find the keypoints with STAR for clockwise 90 degree rotated image
kp_180_clkw = star.detect(img_180_clkw, None)
# compute the descriptors with BRIEF
kp_180_clkw, desc_180_clkw = brief.compute(img_180_clkw, kp_180_clkw)

matches = bf.match(desc1, desc_180_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_180_clkw = cv.drawMatches(img1, kp1, img_90_clkw, kp_180_clkw, matches[300:600], None, flags=2)
cv.imshow('BRIEF: 180 degree clockwise Rotated Image', img3_180_clkw)
cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 180 degree clockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp_180_clkw))
print(len(matches))
#print((len(kp2_180_clkw)/len(kp1))*100)
# ####Image rotate 180 degree clockwise end ###

# ####Image rotate 270 degree clockwise start ###
start = datetime.now()
img_270_clkw = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
# find the keypoints with STAR for clockwise 90 degree rotated image
kp_270_clkw = star.detect(img_270_clkw, None)
# compute the descriptors with BRIEF
kp_270_clkw, desc_270_clkw = brief.compute(img_180_clkw, kp_270_clkw)

matches = bf.match(desc1, desc_270_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_270_clkw = cv.drawMatches(img1, kp1, img_270_clkw, kp_270_clkw, matches[300:600], None, flags=2)
cv.imshow('BRIEF: 90 degree anti-clockwise Rotated Image', img3_270_clkw)
cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 90 degree anticlockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp_270_clkw))
print(len(matches))
#print((len(kp2_270_clkw)/len(kp1))*100)
# ####Image rotate 270 degree clockwise end ###

# ####Image rotate 90 degree clockwise start ###
start = datetime.now()
img_90_clkw = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
'''cv.imshow('BRIEF', img_90_clkw)
cv.waitKey(0)'''

# find the keypoints with STAR for clockwise 90 degree rotated image
kp_90_clkw = star.detect(img_90_clkw, None)
# compute the descriptors with BRIEF
kp_90_clkw, desc_90_clkw = brief.compute(img_90_clkw, kp_90_clkw)

matches = bf.match(desc1, desc_90_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_90_clkw = cv.drawMatches(img1, kp1, img_90_clkw, kp_90_clkw, matches[300:600], img2, flags=2)
cv.imshow('BRIEF: 90 degree clockwise Rotated Image', img3_90_clkw)
cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"TExecution time for 90 degree rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp_90_clkw))
print(len(matches))
#print((len(kp2_90_clkw)/len(kp1))*100)

# ####Image rotate 90 degree clockwise end ###

# ####Image rotate 180 degree clockwise start ###
start = datetime.now()
img_180_clkw = cv.rotate(img2, cv.ROTATE_180)
'''cv.imshow('BRIEF', img_90_clkw)
cv.waitKey(0)'''

# find the keypoints with STAR for clockwise 90 degree rotated image
kp_180_clkw = star.detect(img_180_clkw, None)
# compute the descriptors with BRIEF
kp_180_clkw, desc_180_clkw = brief.compute(img_180_clkw, kp_180_clkw)

matches = bf.match(desc1, desc_180_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_180_clkw = cv.drawMatches(img1, kp1, img_90_clkw, kp_180_clkw, matches[300:600], None, flags=2)
cv.imshow('BRIEF: 180 degree clockwise Rotated Image', img3_180_clkw)
cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 180 degree clockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp_180_clkw))
print(len(matches))
#print((len(kp2_180_clkw)/len(kp1))*100)
# ####Image rotate 180 degree clockwise end ###

# ####Image rotate 270 degree clockwise start ###
start = datetime.now()
img_270_clkw = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
# find the keypoints with STAR for clockwise 90 degree rotated image
kp_270_clkw = star.detect(img_270_clkw, None)
# compute the descriptors with BRIEF
kp_270_clkw, desc_270_clkw = brief.compute(img_180_clkw, kp_270_clkw)

matches = bf.match(desc1, desc_270_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_270_clkw = cv.drawMatches(img1, kp1, img_270_clkw, kp_270_clkw, matches[300:600], None, flags=2)
cv.imshow('BRIEF: 90 degree anti-clockwise Rotated Image', img3_270_clkw)
cv.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 90 degree anticlockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp_270_clkw))
print(len(matches))