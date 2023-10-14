# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 21:23:42 2023

@author: Sarwar
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt


# Load two example images
image1 = cv2.imread('Image_026.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('Image_029.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Create a Brute-Force Matcher
bf = cv2.BFMatcher()

# Match descriptors between the two images
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply a ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Initialize lists to store true positives (TP) and false positives (FP) for different thresholds
thresholds = np.arange(0, 100, 1)
true_positives = []
false_positives = []

# Compute TP and FP rates for each threshold
for threshold in thresholds:
    tp = sum(1 for match in good_matches if match.distance <= threshold)
    fp = sum(1 for match in matches if match.distance <= threshold) - tp
    true_positives.append(tp)
    false_positives.append(fp)

# Calculate TPR and FPR
true_positive_rate = np.array(true_positives) / len(good_matches)
false_positive_rate = np.array(false_positives) / (len(matches) - len(good_matches))

# Plot the ROC-like curve
plt.figure()
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-like Curve for SIFT Feature Matching')
plt.show()
