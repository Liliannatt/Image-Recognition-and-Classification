# EQ2425-Analysis-and-Search-of-Visual-Data

## Project 1

This project focuses on Image Features and Matching, an essential aspect of computer vision. I used keypoint detection and feature matching techniques, specifically SIFT and SURF, to explore their robustness under various transformations like rotation and scaling.

### Objectives
- Keypoint Detect,mion: Apply SIFT and SURF to detect keypoints in images and analyze the effect of changes in the peak and edge thresholds for SIFT and the strongest feature threshold for SURF.

- Repeatability Analysis: Measure and compare the robustness of SIFT and SURF detectors against:
    - Rotation: Vary the angle from 0 to 360 degrees in increments of 15 degrees.
    - Scaling: Modify the image using scaling factors from 1.0 to 1.2^8.

- Image Feature Matching: Implement and analyze three feature matching algorithms:
    - Fixed Threshold Matching
    - Nearest Neighbor Matching
    - Nearest Neighbor Distance Ratio Matching

### Key Features and Deliverables
Keypoint Detection and Visualization:
- Apply SIFT and SURF to detect keypoints in a given image.
- Superimpose detected keypoints onto the original image.
- Report the chosen thresholds for SIFT and SURF, and describe which objects or regions generate numerous keypoints.

Repeatability vs. Transformation Analysis:
- Plot repeatability against rotation angles and scaling factors for both keypoint detectors.
- Compare robustness against transformations like rotation and scaling.

Feature Matching:
- Implement three feature matching algorithms.
- Plot and visually inspect feature matches between pairs of images.
- Compare the performance of SIFT and SURF using the Nearest Neighbor Distance Ratio Matching.
