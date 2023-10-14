# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:32:40 2023

@author: Sarwar
"""

import cv2
from datetime import datetime

img1 = cv2.imread('Image_026.jpg')
img2 = cv2.imread('Image_029.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#Create SIFT Feature Detector object
sift = cv2.SIFT_create()

#Detect key points for fisrt image
kp1 = sift.detect(img1, None)

freakExtractor = cv2.xfeatures2d.FREAK_create()
kp1,desc1 = freakExtractor.compute(img1,kp1)
start = datetime.now()
#Detect key points for 2nd image
kp2 = sift.detect(img2, None)

freakExtractor = cv2.xfeatures2d.FREAK_create()
kp2,desc2 = freakExtractor.compute(img2,kp1)

matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:600], None, flags=2)

cv2.imshow('FREAK: Regular Image', img3)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 2 general image : {td:.03f}ms")
print(len(kp1))
print(len(kp2))
print(len(matches))
print((len(kp2)/len(kp1))*100)

# start 90 degree clockwise rotated image
start = datetime.now()
img2_90_clkw = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
kp2_90_clkw,desc2_90_clkw = freakExtractor.compute(img2_90_clkw,kp1)

matches = bf.match(desc1, desc2_90_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()
img3_90_clkw = cv2.drawMatches(img1, kp1, img2_90_clkw, kp2_90_clkw, matches[0:600], None, flags=2)

cv2.imshow('FREAK: 90 degree clockwise Rotated Image', img3_90_clkw)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 2 general image : {td:.03f}ms")
print(len(kp1))
print(len(kp2_90_clkw))
print(len(matches))
print((len(kp2_90_clkw)/len(kp1))*100)

# start 180 degree clockwise rotated image
start = datetime.now()
img2_180_clkw = cv2.rotate(img2, cv2.ROTATE_180)
kp2_180_clkw,desc2_180_clkw = freakExtractor.compute(img2_180_clkw,kp1)

matches = bf.match(desc1, desc2_180_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()
img3_180_clkw = cv2.drawMatches(img1, kp1, img2_180_clkw, kp2_180_clkw, matches[0:600], None, flags=2)

cv2.imshow('FREAK: 180 degree clockwise Rotated Image', img3_180_clkw)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 2 general image : {td:.03f}ms")
print(len(kp1))
print(len(kp2_180_clkw))
print(len(matches))
print((len(kp2_180_clkw)/len(kp1))*100)
# start 90 degree anticlockwise rotated image
start = datetime.now()
img2_270_clkw = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
kp2_270_clkw,desc2_270_clkw = freakExtractor.compute(img2_270_clkw,kp1)

matches = bf.match(desc1, desc2_270_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()
img3_270_clkw = cv2.drawMatches(img1, kp1, img2_270_clkw, kp2_270_clkw, matches[0:600], None, flags=2)

cv2.imshow('FREAK: 90 degree anti-clockwise Rotated Image', img3_270_clkw)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 2 general image : {td:.03f}ms")
print(len(kp1))
print(len(kp2_270_clkw))
print(len(matches))
print((len(kp2_270_clkw)/len(kp1))*100)