# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
from functools import partial
from itertools import repeat
import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from os import path
import cv2.cv2 as cv2
import torch
from skimage.metrics import structural_similarity as ssim
import cupy as cp
import numpy as np
import time
from multiprocessing import Pool, Process, Manager, freeze_support
import logging
import multiprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# It applies a Gaussian Blur on the images and then calculates the SSIM between the input images
def calc_SSIM(im1_gray, im2_gray):
    im1_gray_gb = cv2.GaussianBlur(im1_gray, (5, 5), cv2.BORDER_DEFAULT)
    im2_gray_gb = cv2.GaussianBlur(im2_gray, (5, 5), cv2.BORDER_DEFAULT)
    ssim_index = ssim(im1_gray_gb, im2_gray_gb)
    return ssim_index


# Given the current frame, it calculates the relative values of a row and a column in the similarity matrix
# Simmetry between row and column values
def similarity_matrix_row_col(frames_list, count, path):
    length = len(frames_list)

    im_count = cv2.imread(os.path.join(os.getcwd(), path, frames_list[count]), 1)
    im_count_gray = cv2.cvtColor(im_count, cv2.COLOR_RGB2GRAY)
    print(count)
    result = cp.zeros(length - count - 1)
    for i, j in enumerate(range(count + 1, length, 1)):
        im_i = cv2.imread(path + '/' + frames_list[j], 1)
        im_i_gray = cv2.cvtColor(im_i, cv2.COLOR_RGB2GRAY)
        sim_index = calc_SSIM(im_i_gray, im_count_gray)
        result[i] = sim_index
        # sim_matrix[i][count] = sim_index
        # sim_matrix[count][i] = sim_index
    return result


# Multiprocessing function to calculate in parallel the values of 4 rows/columns
# It returns the final matrix
def calc_sim_matrix_mp(frames_list, path):
    length = len(frames_list)
    sim_matrix = cp.eye(len(frames_list), length)
    ctx = multiprocessing.get_context("spawn")

    pool = ctx.Pool(processes=4)
    range_id = range(length)
    # res = similarity_matrix(frames_list, 0, path)
    pool_result = pool.starmap(similarity_matrix_row_col, zip(repeat(frames_list), range_id, repeat(path)))
    pool.close()
    pool.join()
    for line, result in enumerate(pool_result):
        sim_matrix[line, line + 1:] = result
        sim_matrix[line + 1:, line] = cp.transpose(result)
    # chuncks = [frames_list[i::4] for i in range(4)]

    # p1 = Process(target=similarity_matrix, args = chuncks[0])
    # p2 = Process(target=similarity_matrix, args = chuncks[1])
    # p3 = Process(target=similarity_matrix, args = chuncks[2])
    # p4 = Process(target=similarity_matrix, args = chuncks[3])

    return sim_matrix


# MAIN FLOW EXECUTION

if __name__ == '__main__':
    freeze_support()
    name_patient_list = os.listdir('processed_frames')   # contains the list of the patients
    name_patient_list.sort()

    name_video_list = []   # contains the list of the videos
    path_video_list = []   # contains the list of the paths of all videos

    for i in name_patient_list:
        if 'Video' not in i:
            name_patient_list.remove(i)

    for i in name_patient_list:
        folder_list = sorted(os.listdir('processed_frames/' + i))

        for elem in folder_list:
            if os.path.isdir(os.path.join(os.getcwd(), 'processed_frames', i, elem)):
                name_video_list.append(elem)
                path_video_list.append('processed_frames/' + i + '/' + elem + '/images')

    for count in range(len(name_video_list)):
        if not path.exists('similarity_matrices/similarity_matrices_gb/' + name_video_list[count] + '.png'):
            frames_list = sorted(os.listdir(path_video_list[count]))   # list of pre-processed images

            sim_matrix = calc_sim_matrix_mp(frames_list, path_video_list[count])
            sim_matrix = cp.asnumpy(sim_matrix)
            #sim_matrix = np.uint8(sim_matrix)

            plt.imshow(sim_matrix, cmap='gray')
            plt.savefig('similarity_matrices_gb/' + name_video_list[count] + '.png')
