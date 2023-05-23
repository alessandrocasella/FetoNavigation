import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def convert_points_to_keypoints(pts, confidence):
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        for i in range(len(pts)):
            keypoint = cv2.KeyPoint(pts[i][0], pts[i][1], size=1, response=confidence[i])
            kps.append(keypoint)
    return kps



mask = cv2.imread(os.path.join(os.getcwd(), '../../final_dataset/anon001/mask.png'), 0)
img_0 = cv2.imread(os.path.join(os.getcwd(), '../../final_dataset/anon001/images/anon001_00912.png'))
img_0 = cv2.cvtColor(img_0 ,cv2.COLOR_BGR2GRAY)
img_1 = cv2.imread(os.path.join(os.getcwd(), '../../final_dataset/anon001/images/anon001_00913.png'))
img_1 = cv2.cvtColor(img_1 ,cv2.COLOR_BGR2GRAY)


# Calculation of eroded mask
erosion_size = 20
erosion_shape = cv2.MORPH_ELLIPSE
element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                    (erosion_size, erosion_size))
mask_eroded = cv2.erode(mask, element)


mask[mask != 0] = 255.
img_0_and = cv2.bitwise_and(img_0, mask)
img_1_and = cv2.bitwise_and(img_1, mask)

# Initializing Sift
sift = cv2.xfeatures2d.SIFT_create()

kp_0, desc_0 = sift.detectAndCompute(img_0_and, None)
kp_1, desc_1 = sift.detectAndCompute(img_1_and, None)


# Initializing Sift
sift = cv2.xfeatures2d.SIFT_create()

kp_0, desc_0 = sift.detectAndCompute(img_0_and, None)
kp_1, desc_1 = sift.detectAndCompute(img_1_and, None)


pts = cv2.KeyPoint_convert(kp_0)

indices = []
for i, elem in enumerate(pts):
    print(elem)
    if (mask[int(elem[0])][int(elem[1])] == 255):
        if (mask_eroded[int(elem[0])][int(elem[1])]!=0):
            indices.append(i)

pts_new = pts[indices]
print(pts)
kp_0 = [cv2.KeyPoint(x[1], x[0], 1) for x in pts_new]


pts = cv2.KeyPoint_convert(kp_1)

indices = []
for i, elem in enumerate(pts):
    print(elem)
    if (mask[int(elem[0])][int(elem[1])] == 255):
        if (mask_eroded[int(elem[0])][int(elem[1])]!=0):
            indices.append(i)

pts_new = pts[indices]
print(pts)
kp_1 = [cv2.KeyPoint(x[1], x[0], 1) for x in pts_new]




img_0_draw = cv2.drawKeypoints(img_0, kp_0, img_0_and)
img_1_draw = cv2.drawKeypoints(img_1, kp_1, img_1_and)


# Using Brute Force matcher

bf = cv2.BFMatcher()

# It tries to find the closest descriptor in the second set , and applyes KNN Matching to the descriptors
matches = bf.knnMatch(desc_0, desc_1, k=2)

# good_matches = []
#
# for i, (m, n) in enumerate(matches):
#     # print(m.queryIdx)
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)
#
# matches_idx = [m.queryIdx for (m, n) in matches]
# kp_0 = [kp_0[idx] for idx in matches_idx]
# desc_0 = [desc_0[idx] for idx in matches_idx]
# matches_idx = [m.trainIdx for (m, n) in matches]
# kp_1 = [kp_1[idx] for idx in matches_idx]
# desc_1 = [desc_1[idx] for idx in matches_idx]

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(good)

img_matches = cv2.drawMatchesKnn(img_0_draw,kp_0,img_1_draw,kp_1,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv2.imwrite('matches.png',img_matches)

concat = np.concatenate((img_0, img_1), axis=1)
cv2.imwrite('concat.png',concat)