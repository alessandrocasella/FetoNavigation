import os
import cv2
import imutils
import numpy as np

path_image = os.path.join(os.getcwd(), '../../final_dataset/anon002/images/anon002_01760.png')

image = cv2.imread(path_image, 1)

# line to apply rotation
# image_rot = imutils.rotate(image, 30)
# cv2.imwrite('test_rotation.png',image_rot)

# line to apply illumination changes
image_contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=1)
cv2.imwrite('test_contrast.png',image_contrast)

# line to apply illumination changes
image_brightness = cv2.convertScaleAbs(image, alpha=1, beta=-30)
cv2.imwrite('test_brightness.png',image_brightness)

# lines to apply scaling
# shape = image.shape[0]
# image_crop = image[50:image.shape[1] - 50, 50:image.shape[0] - 50]
# image_crop = cv2.resize(image_crop, (shape, shape))
# cv2.imwrite('test_crop.png',image_crop)

# lines to add a black patch in the centre
# image_copy = np.copy(image)
# patch = np.zeros((100, 100, 3), dtype='uint8')
# x_offset = (image.shape[0] - patch.shape[0]) // 2
# y_offset = (image.shape[1] - patch.shape[1]) // 2
# image_copy[y_offset:y_offset + patch.shape[1], x_offset:x_offset + patch.shape[0]] = patch
# cv2.imwrite('test_patch.png',image_copy)