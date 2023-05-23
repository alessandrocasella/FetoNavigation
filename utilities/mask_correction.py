import cv2
import os
import numpy as np

name_video = 'anon005'
mask = cv2.imread(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'mask.png'), 0)
mask = mask/254.
mask = mask*255.
#print(np.max(mask))
cv2.imwrite(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'mask.png'), mask)
