# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2.cv2 as cv2
import numpy as np


name_video = 'Video013'
path_folder = os.path.join(os.getcwd(), 'processed_frames/Video013')

list_folders_names = sorted(os.listdir(path_folder))
for i in list_folders_names:
    if ('Video' not in i):
        list_folders_names.remove(i)

for elem in list_folders_names:
    list_frames = []
    path_frames = os.path.join(path_folder, elem, 'images')
    list_frames_names = sorted(os.listdir(path_frames))

    new_cut_frames_path = os.path.join(path_folder, elem, 'images_new_cut')

    try:
        os.mkdir(new_cut_frames_path)
    except:
        print('error')

    for i in list_frames_names:
        print(i)
        image = cv2.imread(path_frames+'/'+i,1)
        width = image.shape[1]
        height = image.shape[0]
        x_left = (width - height) // 2
        x_right = width - x_left

        im_squared = image[0:height, x_left-10:x_right-10]
        cv2.imwrite(new_cut_frames_path+'/'+i, im_squared)
