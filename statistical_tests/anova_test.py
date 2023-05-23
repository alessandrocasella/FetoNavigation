from scipy.stats import median_test
import os
import numpy as np


video_names = sorted(os.listdir(os.path.join(os.getcwd(), '../final_dataset')))
for elem in video_names:
    if ('.DS' in elem):
        video_names.remove(elem)

for video_name in video_names:

    path_data_1 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', 'Bano_mosaicking', 'ssim_mosaicking_' + video_name + '.npy')
    path_data_2 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', 'SLAM_SIFT', 'ssim_mosaicking_' + video_name + '.npy')
    path_data_3 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', 'SLAM_ORB', 'ssim_mosaicking_' + video_name + '.npy')
    path_data_4 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', 'SLAM_LoFTR_FINAL', 'ssim_mosaicking_' + video_name + '.npy')

    paths = [path_data_1, path_data_2, path_data_3, path_data_4]
    shape = 0
    for path in paths:
        if (os.path.exists(path)):
            data_shape = np.load(path)
            shape = data_shape.shape[0]

    arr = np.zeros((shape, 4), dtype=float)

    for index,path in enumerate(paths):
        try:
            data = np.load(path)

            if (data.shape[0] != shape):
                arr = arr[:-1]
            arr[:, index] = data[:arr.shape[0]]
        except:
            pass


    np.savetxt("ssim_"+video_name+".txt", arr, fmt='%.5f',delimiter=";")