# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2.cv2 as cv2
import numpy as np



def get_min_shape(list_frames_names, path):
    min_shape0 = 5000
    min_shape1 = 5000

    for elem in list_frames_names:
        path_frames = os.path.join(path, elem)
        print(path_frames)
        image = cv2.imread(path_frames, 0)
        print(image.shape)
        if (image.shape[0] < min_shape0):
            min_shape0 = image.shape[0]
        if (image.shape[1] < min_shape1):
            min_shape1 = image.shape[1]
    print(min_shape0)
    print(min_shape1)
    return min_shape0, min_shape1


name_video = 'anon016'
path_folder = os.path.join(os.getcwd(), 'processed_frames/anon016/Video016_CLIP30/images')

list_frames_names = sorted(os.listdir(path_folder))
print(list_frames_names)
# for i in list_frames_names:
#     if ('Video' not in i):
#         list_frames_names.remove(i)

min_shape0, min_shape1 = get_min_shape(list_frames_names, path_folder)

# new_cut_frames_path = os.path.join(os.getcwd(), 'processed_frames/anon016/Video016_CLIP30', 'images_new_cut')
#
# try:
#     os.mkdir(new_cut_frames_path)
# except:
#     print('error')
#
#
# for elem in list_frames_names:
#     im_squared = image[0:min_shape1, 0:min_shape0]
#     cv2.imwrite(new_cut_frames_path+'/'+elem, im_squared)