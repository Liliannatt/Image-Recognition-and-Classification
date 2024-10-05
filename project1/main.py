import cv2
import matplotlib.pyplot as plt
import numpy as np

### 2 Robustness of Keypoint Detectors
### 2.1 Read and Load the images(3a)
image1_path = 'data1\data1\obj1_5.JPG'
image2_path = 'data1\data1\obj1_t1.JPG'
img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Show the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Image 1')

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title('Image 2')

plt.show()

### 2.2 SIFT feature extraction
# SIFT Detector
# Initialize the SIFT detector
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.1, edgeThreshold=10)

# Detect keypoints and descriptors
keypoints_sift_1, descriptors_sift_1 = sift.detectAndCompute(img1, None)
keypoints_sift_2, descriptors_sift_2 = sift.detectAndCompute(img2, None)

# Draw keypoints on the images
img1_sift = cv2.drawKeypoints(img1, keypoints_sift_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_sift = cv2.drawKeypoints(img2, keypoints_sift_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show the keypoints
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_sift)
plt.title('SIFT Keypoints - Image 1')

plt.subplot(1, 2, 2)
plt.imshow(img2_sift)
plt.title('SIFT Keypoints - Image 2')
cv2.imwrite('output\sift_keypoints1.jpg', img1_sift)
cv2.imwrite('output\sift_keypoints2.jpg', img2_sift)
plt.show()
print(f"Number of SIFT keypoints1 detected: {len(keypoints_sift_1)}")
print(f"Number of SIFT keypoints2 detected: {len(keypoints_sift_2)}")


### 2.3 SURF Detector
# Initialize the SURF detector
surf = cv2.xfeatures2d.SURF_create(5000)

# Detect keypoints and descriptors
keypoints_surf_1, descriptors_surf_1 = surf.detectAndCompute(img1, None)
keypoints_surf_2, descriptors_surf_2 = surf.detectAndCompute(img2, None)

# Draw keypoints on the images
img1_surf = cv2.drawKeypoints(img1, keypoints_surf_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_surf = cv2.drawKeypoints(img2, keypoints_surf_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show the keypoints
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_surf)
plt.title('SURF Keypoints - Image 1')

plt.subplot(1, 2, 2)
plt.imshow(img2_surf)
plt.title('SURF Keypoints - Image 2')

cv2.imwrite('output\surf_keypoints1.jpg', img1_surf)
cv2.imwrite('output\surf_keypoints2.jpg', img2_surf)
plt.show()
print(f"Number of SURF keypoints1 detected: {len(keypoints_surf_1)}")
print(f"Number of SURF keypoints2 detected: {len(keypoints_surf_2)}")


## 2.4 Rotation Robustness(plot repeatability vs rotation angle)
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Function to compute repeatability
def compute_repeatability(kp1, kp2, img1_shape, img2_shape):
    # Use FLANN-based matcher to match keypoints between two images
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(kp1, kp2)
    # Repeatability = Number of matches / Number of keypoints in original image
    return len(matches) / min(len(kp1), len(kp2))


# Rotate the image in increments of 15 degrees
angles = range(0, 360, 15)
repeatability_sift = []
repeatability_surf = []

for angle in angles:
    rotated_image = rotate_image(img1, angle)
    keypoints_sift_rot, descriptors_sift_rot = sift.detectAndCompute(rotated_image, None)
    keypoints_surf_rot, descriptors_surf_rot = surf.detectAndCompute(rotated_image, None)
    
    # Compute repeatability as ratio of matching keypoints
    repeatability_sift.append(compute_repeatability(descriptors_sift_1, descriptors_sift_rot, img1.shape, rotated_image.shape))
    repeatability_surf.append(compute_repeatability(descriptors_surf_1, descriptors_surf_rot, img1.shape, rotated_image.shape))

    

# Plot rotation repeatability
plt.plot(angles, repeatability_sift, label='SIFT Repeatability')
plt.plot(angles, repeatability_surf, label='SURF Repeatability')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Repeatability')
plt.title('Repeatability vs Rotation Angle')
plt.legend()
plt.savefig('output/output_repeatability_vs_rotation_fixed.jpg')
plt.show()


## 2.5. Scale Robustness(plot repeatability vs scaling factor)

def scale_image(image, scale_factor):
    return cv2.resize(img1, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

scaling_factors = [1.2 ** i for i in range(0, 9)]  
repeatability_sift_scale = []
repeatability_surf_scale = []

for scale in scaling_factors:
    scaled_image = scale_image(img1, scale)
    kp_sift_scale, desc_sift_scale = sift.detectAndCompute(scaled_image, None)
    kp_surf_scale, desc_surf_scale = surf.detectAndCompute(scaled_image, None)
    

    # Compute repeatability as ratio of matching keypoints
    repeatability_sift_scale.append(compute_repeatability(descriptors_sift_1, desc_sift_scale, img1.shape, scaled_image.shape))
    repeatability_surf_scale.append(compute_repeatability(descriptors_surf_1, desc_surf_scale, img1.shape, scaled_image.shape))

# Plot repeatability vs scaling factor
plt.plot(scaling_factors, repeatability_sift_scale, label='SIFT Repeatability')
plt.plot(scaling_factors, repeatability_surf_scale, label='SURF Repeatability')
plt.xlabel('Scaling Factor')
plt.ylabel('Repeatability')
plt.title('Repeatability vs Scaling Factor')
plt.legend()
plt.savefig('output/output_repeatability_vs_scaling_fixed.jpg')
plt.show()





#################################################
### 3 Image Feature Matching
# 3(b) Fixed Threshold Matching
# BFMatcher with L2 norm and crossCheck enabled
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors_sift_1, descriptors_sift_2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Set threshold for best and suboptimal matches
best_threshold = 100
suboptimal_threshold = 150

# Filter matches based on thresholds
best_matches = [m for m in matches if m.distance < best_threshold]
suboptimal_matches = [m for m in matches if best_threshold <= m.distance < suboptimal_threshold]

# Visualize the matches
img_matches = cv2.drawMatches(img1, keypoints_sift_1, img2, keypoints_sift_2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 5))
plt.imshow(img_matches)
plt.title('Fixed Threshold Matching - Top 50 Matches')
plt.show()

# Save the match results
cv2.imwrite('output/fixed_threshold_matches.jpg', img_matches)

print(f"Number of best matches: {len(best_matches)}")
print(f"Number of suboptimal matches: {len(suboptimal_matches)}")

#3(c) Nearest Neighbor Matching
# Use BFMatcher for Nearest Neighbor Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches_nn = bf.match(descriptors_sift_1, descriptors_sift_2)

# Sort them in the order of their distance
matches_nn = sorted(matches_nn, key=lambda x: x.distance)

# Draw the first 50 matches
img_matches_nn = cv2.drawMatches(img1, keypoints_sift_1, img2, keypoints_sift_2, matches_nn[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches
plt.figure(figsize=(15, 5))
plt.imshow(img_matches_nn)
plt.title('Nearest Neighbor Matching using SIFT')
plt.show()

# Save the matched image
cv2.imwrite('output\sift_nearest_neighbor_matches.jpg', img_matches_nn)


# 3(d) sift feature matching
# Use FLANN-based matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Perform Nearest Neighbor Distance Ratio Matching
matches = flann.knnMatch(descriptors_sift_1, descriptors_sift_2, k=2)

# Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good_matches.append(m)

# Draw the matches
img_matches_ratio = cv2.drawMatches(img1, keypoints_sift_1, img2, keypoints_sift_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches
plt.figure(figsize=(15, 5))
plt.imshow(img_matches_ratio)
plt.title('Nearest Neighbor Distance Ratio Matching using SIFT')
plt.show()

# Save the results
cv2.imwrite('output/nn_distance_ratio_matches.jpg', img_matches_ratio)




# 3(e) SURF Feature Extraction and Nearest Neighbor Distance Ratio Matching
# FLANN-based matcher for SURF
matches_surf = flann.knnMatch(descriptors_surf_1, descriptors_surf_2, k=2)

# Apply the ratio test
good_matches_surf = []
for m, n in matches_surf:
    if m.distance < 0.6 * n.distance:
        good_matches_surf.append(m)

# Draw matches
img_matches_surf = cv2.drawMatches(img1, keypoints_surf_1, img2, keypoints_surf_2, good_matches_surf, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches
plt.figure(figsize=(15, 5))
plt.imshow(img_matches_surf)
plt.title('SURF Nearest Neighbor Distance Ratio Matching')
plt.show()

# Save the results
cv2.imwrite('output/surf_nn_distance_ratio_matches.jpg', img_matches_surf)
