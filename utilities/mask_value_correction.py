import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
path_image = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/mask.png')
img = cv2.imread(path_image, 1)
# img[img != 0] = 255
#
print(np.unique(img))
# cv2.imwrite(path_image, img)
