# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:40:15 2023

@author: Sarwar
This is first version
"""

import cv2
from datetime import datetime
import numpy as np
import math

# sift
sift = cv2.SIFT_create()

# feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# img1 = cv2.imread('input_image.jpg')
# img2 = cv2.imread('input_image.jpg')
# img1 = cv2.imread('testimg1.jpg')
# img2 = cv2.imread('testimg2.jpg')

# img1 = cv2.imread('Image_298.jpg')
# img2 = cv2.imread('Image_300.jpg')

img1 = cv2.imread('Image_026.jpg')
img2 = cv2.imread('Image_029.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, desc1 = sift.detectAndCompute(img1, None)
start = datetime.now()
kp2, desc2 = sift.detectAndCompute(img2, None)

matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:300], None, flags=2)

cv2.imshow('SIFT Regular Image', img3)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3

# Set a distance threshold for consistency
consistency_threshold = 30


# Get consistently matched keypoints
consistent_matches = [match for match in matches if match.distance < consistency_threshold]

# Calculate detection rate
detection_rate = (len(consistent_matches) / max(len(kp1), len(kp2))) * 100

print("Detection Rate:", detection_rate)

# Calculate false positives
false_positive_count = max(len(kp1), len(kp2)) - len(consistent_matches)

# Calculate false positive rate
false_positive_rate = (false_positive_count / max(len(kp1), len(kp2))) * 100

print("False Positive Rate:", false_positive_rate)

# Calculate matching robustness
matching_robustness = (len(consistent_matches) / max(len(kp1), len(kp2))) * 100

print("Matching Robustness:", matching_robustness)

# Calculate repeatability
repeatability = (len(consistent_matches) / len(kp2)) * 100

print("Repeatability:", repeatability)

# Calculate matching accuracy
correctly_matched_count = sum(1 for match in consistent_matches if match.queryIdx == match.trainIdx)
matching_accuracy = (correctly_matched_count / len(consistent_matches)) * 100

print("Correctly Match count:", correctly_matched_count)
print("Matching Accuracy:", matching_accuracy)

# Get consistently matched keypoints
# =============================================================================
# consistent_matches = [match for match in matches if match.distance < consistency_threshold]
# consistent_keypoints1 = np.array([kp1[match.queryIdx].pt for match in consistent_matches])
# consistent_keypoints2 = np.array([kp2[match.trainIdx].pt for match in consistent_matches])
# 
# # Calculate localization errors
# localization_errors = np.sqrt(np.sum((consistent_keypoints1 - consistent_keypoints2)**2, axis=1))
# # Compute statistics
# mean_error = np.mean(localization_errors)
# median_error = np.median(localization_errors)
# std_deviation = np.std(localization_errors)
# 
# print("Mean Localization Error:", mean_error)
# print("Median Localization Error:", median_error)
# print("Standard Deviation of Localization Errors:", std_deviation)
# =============================================================================

print(f"Execution time for 2 general image : {td:.03f}ms")
print(len(kp1))
print(len(kp2))
print(len(matches))
print("Consistence Match", len(consistent_matches))
print("Consistence Match", consistent_matches)
print((len(kp2)/len(kp1))*100)

###clockwise 90 rotated
start = datetime.now()
img2_90_clkw = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
kp2_90_clkw, desc2_90_clkw = sift.detectAndCompute(img2_90_clkw, None)

matches = bf.match(desc1, desc2_90_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_90_clkw = cv2.drawMatches(img1, kp1, img2_90_clkw, kp2_90_clkw, matches[0:1000], None, flags=2)

cv2.imshow('SIFT: 90 degree clockwise Rotated Image', img3_90_clkw)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 90 degree clockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp2_90_clkw))
print(len(matches))
print((len(kp2_90_clkw)/len(kp1))*100)

# Set a distance threshold for consistency
#consistency_threshold = 30

# Count consistent keypoints
consistent_matches = [match for match in matches if match.distance < consistency_threshold]
consistent_keypoints = set([match.queryIdx for match in consistent_matches])

# Calculate percentage of consistent keypoints
percentage_consistent = (len(consistent_keypoints) / len(kp1)) * 100

print("Percentage of Consistent Keypoints:", percentage_consistent)

###clockwise 180 rotated
start = datetime.now()
img2_180_clkw = cv2.rotate(img2,  cv2.ROTATE_180)
kp2_180_clkw, desc2_180_clkw = sift.detectAndCompute(img2_180_clkw, None)

matches = bf.match(desc1, desc2_180_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_180_clkw = cv2.drawMatches(img1, kp1, img2_180_clkw, kp2_180_clkw, matches[0:1000], None, flags=2)

cv2.imshow('SIFT: 180 degree clockwise Rotated Image', img3_180_clkw)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 180 degree clockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp2_180_clkw))
print(len(matches))
print((len(kp2_180_clkw)/len(kp1))*100)

# Count consistent keypoints
consistent_matches = [match for match in matches if match.distance < consistency_threshold]
consistent_keypoints = set([match.queryIdx for match in consistent_matches])

# Calculate percentage of consistent keypoints
percentage_consistent = (len(consistent_keypoints) / len(kp1)) * 100

print("Percentage of Consistent Keypoints:", percentage_consistent)

###anti-clockwise 90 rotated
start = datetime.now()
img2_270_clkw = cv2.rotate(img2,  cv2.ROTATE_90_COUNTERCLOCKWISE)
kp2_270_clkw, desc2_270_clkw = sift.detectAndCompute(img2_270_clkw, None)

matches = bf.match(desc1, desc2_270_clkw)
matches = sorted(matches, key=lambda x: x.distance)
end = datetime.now()

img3_270_clkw = cv2.drawMatches(img1, kp1, img2_270_clkw, kp2_270_clkw, matches[0:1000], None, flags=2)

cv2.imshow('SIFT: 90 degree anti-clockwise Rotated Image', img3_270_clkw)

cv2.waitKey(0)
td = (end - start).total_seconds() * 10**3
print(f"Execution time for 90 degree anti-clockwise rotated image : {td:.03f}ms")
print(len(kp1))
print(len(kp2_270_clkw))
print(len(matches))
print((len(kp2_270_clkw)/len(kp1))*100)

# Count consistent keypoints
consistent_matches = [match for match in matches if match.distance < consistency_threshold]
consistent_keypoints = set([match.queryIdx for match in consistent_matches])

# Calculate percentage of consistent keypoints
percentage_consistent = (len(consistent_keypoints) / len(kp1)) * 100

print("Percentage of Consistent Keypoints:", percentage_consistent)

##Localization Error
# Set a distance threshold for consistency
#consistency_threshold = 30

# Get consistently matched keypoints
consistent_matches = [match for match in matches if match.distance < consistency_threshold]
consistent_keypoints1 = np.array([kp1[match.queryIdx].pt for match in consistent_matches])
consistent_keypoints2 = np.array([kp2[match.trainIdx].pt for match in consistent_matches])

# Calculate localization errors
localization_errors = np.sqrt(np.sum((consistent_keypoints1 - consistent_keypoints2)**2, axis=1))

# Compute statistics
mean_error = np.mean(localization_errors)
median_error = np.median(localization_errors)
std_deviation = np.std(localization_errors)

print("Mean Localization Error:", mean_error)
print("Median Localization Error:", median_error)
print("Standard Deviation of Localization Errors:", std_deviation)