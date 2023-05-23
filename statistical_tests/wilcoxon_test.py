import os
import numpy as np
from scipy.stats import wilcoxon, ranksums, kruskal #, median_test

list_tries = ['Bano_mosaicking', 'SLAM_SIFT', 'SLAM_ORB', 'SLAM_LoFTR_FINAL']
videos_1 = ['anon001', 'anon002','anon003','anon005', 'anon010']
videos_2 = ['anon012','anon016','anon016_CLIP18', 'anon024_CLIP00', 'anon024_CLIP01']


video_name = 'anon024_CLIP01'
print(video_name)
#VIDEO 1
name_try_1 = 'Bano_mosaicking'
name_try_2 = 'SLAM_SIFT'

path_data_1 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_1, 'ssim_mosaicking_' + video_name + '.npy')
path_data_2 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_2, 'ssim_mosaicking_' + video_name + '.npy')

data_1 = np.load(path_data_1)
#data_2 = np.load(path_data_2)
data_2 = np.zeros_like(data_1)


w, p = wilcoxon(data_1, data_2)
print(name_try_1, name_try_2, p)


name_try_1 = 'Bano_mosaicking'
name_try_2 = 'SLAM_ORB'

path_data_1 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_1, 'ssim_mosaicking_' + video_name + '.npy')
path_data_2 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_2, 'ssim_mosaicking_' + video_name + '.npy')

data_1 = np.load(path_data_1)
data_2 = np.load(path_data_2)
data_2 = data_2[:-1]

w, p = wilcoxon(data_1, data_2)
print(name_try_1, name_try_2, p)


name_try_1 = 'Bano_mosaicking'
name_try_2 = 'SLAM_LoFTR_FINAL'

path_data_1 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_1, 'ssim_mosaicking_' + video_name + '.npy')
path_data_2 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_2, 'ssim_mosaicking_' + video_name + '.npy')

data_1 = np.load(path_data_1)
data_2 = np.load(path_data_2)
data_2 = data_2[:-1]

w, p= wilcoxon(data_1, data_2)
print(name_try_1, name_try_2, p)


name_try_1 = 'SLAM_SIFT'
name_try_2 = 'SLAM_ORB'

path_data_1 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_1, 'ssim_mosaicking_' + video_name + '.npy')
path_data_2 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_2, 'ssim_mosaicking_' + video_name + '.npy')

#data_1 = np.load(path_data_1)
data_1 = np.zeros_like(data_2)
data_2 = np.load(path_data_2)
data_2 = data_2[:-1]

w, p= wilcoxon(data_1, data_2)
print(name_try_1, name_try_2, p)

name_try_1 = 'SLAM_SIFT'
name_try_2 = 'SLAM_LoFTR_FINAL'

path_data_1 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_1, 'ssim_mosaicking_' + video_name + '.npy')
path_data_2 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_2, 'ssim_mosaicking_' + video_name + '.npy')

#data_1 = np.load(path_data_1)
data_1 = np.zeros_like(data_2)
data_2 = np.load(path_data_2)
data_2 = data_2[:-1]

w, p = wilcoxon(data_1, data_2)
print(name_try_1, name_try_2, p)


name_try_1 = 'SLAM_ORB'
name_try_2 = 'SLAM_LoFTR_FINAL'

path_data_1 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_1, 'ssim_mosaicking_' + video_name + '.npy')
path_data_2 = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try_2, 'ssim_mosaicking_' + video_name + '.npy')

data_1 = np.load(path_data_1)
data_2 = np.load(path_data_2)


w, p = wilcoxon(data_1, data_2)
print(name_try_1, name_try_2, p)