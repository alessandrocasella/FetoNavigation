# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
from functools import partial
from itertools import repeat
import matplotlib
#from tqdm import tqdm
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from os import path
import cv2.cv2 as cv2
import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time
from multiprocessing import Pool, Process, Manager, freeze_support
import logging
import multiprocessing



# It applies a Gaussian Blur on the images and then calculates the SSIM between the input images
def calc_SSIM(im1_gray, im2_gray):
    im1_gray = cv2.GaussianBlur(im1_gray, (9, 9), cv2.BORDER_DEFAULT)
    im2_gray = cv2.GaussianBlur(im2_gray, (9, 9), cv2.BORDER_DEFAULT)
    ssim_index = ssim(im1_gray, im2_gray)
    return ssim_index


# Given the current frame, it calculates the relative values of a row and a column in the similarity matrix
# Simmetry between row and column values
def similarity_matrix_row_col(frames_list, count, path):
    length = len(frames_list)
    im_count = cv2.imread(os.path.join(os.getcwd(), path, frames_list[count]), 1)
    im_count_gray = cv2.cvtColor(im_count, cv2.COLOR_RGB2GRAY)
    print(count)
    result = np.zeros(length - count - 1)
    for i, j in enumerate(range(count + 1, length, 1)):
        im_i = cv2.imread(path + '/' + frames_list[j], 1)
        im_i_gray = cv2.cvtColor(im_i, cv2.COLOR_RGB2GRAY)
        sim_index = calc_SSIM(im_i_gray, im_count_gray)
        result[i] = sim_index
        # sim_matrix[i][count] = sim_index
        # sim_matrix[count][i] = sim_index
    return result


# It creates a vector of length equal to the buffer window where the values are the values of the similarity matrix
# of the given frame (the most recent frame to add to the similarity matrix)
def add_new_row_col(frames_list, buffer_windows, num_current_frame):
    result = np.zeros(buffer_windows-1)
    im_count = cv2.imread(os.path.join(os.getcwd(), path, frames_list[num_current_frame+25-1]), 1)
    im_count_gray = cv2.cvtColor(im_count, cv2.COLOR_RGB2GRAY)

    j = 0
    for i in range(num_current_frame-25, num_current_frame+25-1, 1):
        im_i = cv2.imread(path + '/' + frames_list[i], 1)
        im_i_gray = cv2.cvtColor(im_i, cv2.COLOR_RGB2GRAY)
        sim_index = calc_SSIM(im_i_gray, im_count_gray)
        result[j] = sim_index
        j = j+1

    return result


# It creates a vector of length equal to the buffer window where the values are the values of the similarity matrix
# of the given frame (the most recent frame to add to the similarity matrix)
def calc_sim_matrix_mp(frames_list, path):
    length = len(frames_list)
    sim_matrix = np.eye(len(frames_list), length)
    ctx = multiprocessing.get_context("spawn")

    pool = ctx.Pool(processes=4)
    range_id = range(length)
    # res = similarity_matrix(frames_list, 0, path)
    pool_result = pool.starmap(similarity_matrix_row_col, zip(repeat(frames_list), range_id, repeat(path)))
    pool.close()
    pool.join()
    for line, result in enumerate(pool_result):
        sim_matrix[line, line + 1:] = result
        sim_matrix[line + 1:, line] = np.transpose(result)
    return sim_matrix


# Multiprocessing function to calculate in parallel the values of 4 rows/columns
# It returns the matrix that is used as initial matrix
def similarity_matrix_rt(frames_list, path, buffer_windows, shape, name_video):
    frames_list_buffer = frames_list[0:buffer_windows]
    num_frame_buffer = buffer_windows//2
    sim_matrix_buffer = calc_sim_matrix_mp(frames_list_buffer, path)
    current_map = sim_matrix_buffer

    fps = 25.0
    video_path = os.path.join(os.getcwd(), 'similarity_matrices_rt_gb_9', name_video)
    video_matrix = cv2.VideoWriter(video_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (shape*2, shape))

    current_map_to_plot = cv2.resize(sim_matrix_buffer, (shape, shape), interpolation=cv2.INTER_NEAREST)
    max = np.max(current_map_to_plot)
    min = np.min(current_map_to_plot)
    current_map_to_plot_scaled = np.array([(x-min)/(max-min) for x in current_map_to_plot])
    img = cv2.imread(path + '/' + frames_list[num_frame_buffer], 1)
    img= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    np_horiz_concat = np.concatenate((img, current_map_to_plot_scaled*255), axis=1)
    np_horiz_concat = np.uint8(np_horiz_concat)
    np_horiz_concat = cv2.cvtColor(np_horiz_concat, cv2.COLOR_GRAY2RGB)
    video_matrix.write(np_horiz_concat)

    #for count in range(0,len(frames_list), 1):
    for count in range(num_frame_buffer,len(frames_list)-num_frame_buffer, 1):
        print(count)
        current_map = current_map[1:, 1:]
        result = add_new_row_col(frames_list, buffer_windows, count)
        current_map = np.column_stack([current_map, result])
        result_t = np.transpose(result)
        result_t = np.append(result_t, 1)
        current_map = np.vstack((current_map, result_t))

        current_map_to_plot = cv2.resize(current_map, (shape, shape), interpolation=cv2.INTER_NEAREST)
        max = np.max(current_map_to_plot)
        min = np.min(current_map_to_plot)
        current_map_to_plot_scaled = np.array([(x - min) / (max - min) for x in current_map_to_plot])
        img = cv2.imread(path + '/' + frames_list[count], 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        np_horiz_concat = np.concatenate((img, current_map_to_plot_scaled * 255), axis=1)
        np_horiz_concat = np.uint8(np_horiz_concat)
        np_horiz_concat = cv2.cvtColor(np_horiz_concat, cv2.COLOR_GRAY2RGB)
        video_matrix.write(np_horiz_concat)

    video_matrix.release()
    cv2.destroyAllWindows()



# MAIN FLOW EXECUTION

if __name__ == '__main__':
    freeze_support()
    name_patient_list = sorted(os.listdir('processed_frames'))   # contains the list of the patients

    name_video_list = []   # contains the list of the videos
    path_video_list = []   # contains the list of the paths of all videos

    for i in name_patient_list:
        if ('Video' not in i):
            name_patient_list.remove(i)

    for i in name_patient_list:
        folder_list = sorted(os.listdir('processed_frames/' + i))

        for elem in folder_list:
                if os.path.isdir(os.path.join(os.getcwd(), 'processed_frames', i, elem)):
                    name_video_list.append(elem)
                    path_video_list.append('processed_frames/' + i + '/' + elem + '/images')

    for count in range(len(name_video_list)):
        if not os.path.exists(os.path.join(os.getcwd(), 'similarity_matrices/similarity_matrices_rt_gb_9/', name_video_list[count] + '.mp4')):
            path = path_video_list[count]
            name_video = name_video_list[count]
            print(name_video)
            frames_list = sorted(os.listdir(path_video_list[count]))   # list of pre-processed images
            for i in frames_list:
                if ('Video' not in i):
                    frames_list.remove(i)
            img = cv2.imread(path_video_list[count] + '/' + frames_list[0], 1)
            shape = img.shape[0]
            buffer_windows = 50
            similarity_matrix_rt(frames_list, path, buffer_windows, shape, name_video)