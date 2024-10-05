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


## Project 2

This project focuses on building a Visual Search System that recognizes building objects using SIFT descriptors and vocabulary trees. The visual search system retrieves the objects that are most similar to the query object by using TF-IDF (Term Frequency-Inverse Document Frequency) scoring for retrieval.

### Key Components

Image Feature Extraction
- Extract thousands of SIFT features from database images and query images.
- Index features according to the object number and report the average number of features extracted.

Vocabulary Tree Construction
- Hierarchical k-means Algorithm: Use the hierarchical k-means algorithm to build the vocabulary tree, with control over:
    Branch Number (b): The number of branches each node has.
    Tree Depth: The number of levels in the tree.
- Store necessary information in tree nodes, including data needed for querying by SIFT features and TF-IDF information in the leaf nodes.
- Function used: hi_kmeans(data, b, depth) to generate the vocabulary tree, where:
    data holds SIFT features from the database objects.
    b is the branch number for each level.
    depth is the number of levels.

Querying
- Use the constructed vocabulary tree to send descriptors of each query object and rank the database objects based on TF-IDF scores.
- Test with 50 query objects and calculate the average recall rate:
-   Recall is object-based: It is either 0 or 1 per object.
- Experiment with different vocabulary tree configurations:
    Settings: b = 4, depth = 3, b = 4, depth = 5, b = 5, depth = 7.
    Report average top-1 and top-5 recall rates for each setting.
- Further experiments using the b = 5, depth = 7 tree:
    Query with reduced numbers of query features (90%, 70%, or 50%).
    Report the average top-1 and top-5 recall rates.