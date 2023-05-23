import os
from copy import deepcopy
from tqdm import trange
import kornia
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=sys.maxsize)
#import pydevd_pycharm
from datetime import datetime
from torch.multiprocessing import Manager
import torch.multiprocessing as mp
from code_Git_imported.LoFTR.src.loftr import LoFTR, default_cfg

# Global variables
NEW_SHAPE = 448
DIS_THRESHOLD = 100
CONFIDENCE_THRESHOLD = 0.75
PERCENTAGE = 0.10
NUM_POINTS_THRESHOLD = 70
CANVAS_SHAPE = 4000
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
name_experiment = 'SLAM_LoFTR_simulation'  # string containing the name used to identify the experiment
DATASET_PATH = os.path.join(os.getcwd(), 'final_dataset')
DATASET_FILES_PATH = os.path.join(os.getcwd(), 'final_dataset_files')

if not os.path.exists(DATASET_FILES_PATH):
    os.makedirs(DATASET_FILES_PATH)


path_image_transparent_folder = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment, 'transparent_bg')
if not os.path.exists(path_image_transparent_folder):
    os.makedirs(path_image_transparent_folder)



canvas_center = np.array(
    [[np.cos(0), np.sin(0), CANVAS_SHAPE / 2], [-np.sin(0), np.cos(0), CANVAS_SHAPE / 2], [0, 0, 1]])
img_origin = np.array([[np.cos(0), np.sin(0), -NEW_SHAPE / 2], [-np.sin(0), np.cos(0), -NEW_SHAPE / 2], [0, 0, 1]])
INITIAL_MATRIX = img_origin @ canvas_center




def LoFTR_pair(image_pair, mask, matcher, index_iteration, list_frames, seq_is_end):
    img0_raw = image_pair[0]
    img1_raw = image_pair[1]

    img0_raw = cv2.resize(img0_raw, (NEW_SHAPE, NEW_SHAPE))
    img1_raw = cv2.resize(img1_raw, (NEW_SHAPE, NEW_SHAPE))
    mask[mask != 0] = 255.
    img0_raw = cv2.bitwise_and(img0_raw, mask)
    img1_raw = cv2.bitwise_and(img1_raw, mask)

    img0 = torch.from_numpy(img0_raw)[None][None].to(device) / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].to(device) / 255.
    img0 = kornia.filters.gaussian_blur2d(img0, (9, 9), (1.5, 1.5))
    img1 = kornia.filters.gaussian_blur2d(img1, (9, 9), (1.5, 1.5))
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction all in pytorch
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f']
        mkpts1 = batch['mkpts1_f']
        mconf = batch['mconf']
        descriptors_img0_tensor = matcher.des0
        descriptors_img1_tensor = matcher.des1
        # feature space 3136x256
        feature_space0 = matcher.feat_c0
        feature_space1 = matcher.feat_c1

    descriptors_img0 = descriptors_img0_tensor
    descriptors_img1 = descriptors_img1_tensor

    img0_raw = img0.squeeze()
    img1_raw = img1.squeeze()
    # img1_raw = img1_raw.cpu().detach().numpy()

    # Delete points with a low confidence
    mconf_new = mconf < CONFIDENCE_THRESHOLD
    indices = mconf_new.nonzero().squeeze().to(device)
    mconf = torch.index_select(mconf, 0, indices)
    mkpts0 = torch.index_select(mkpts0, 0, indices)
    mkpts1 = torch.index_select(mkpts1, 0, indices)
    descriptors_img0 = torch.index_select(descriptors_img0, 0, indices)
    descriptors_img1 = torch.index_select(descriptors_img1, 0, indices)

    # Calculation of eroded mask
    erosion_size = 10
    erosion_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    mask_eroded = cv2.erode(mask, element)

    mask_eroded_tensor = torch.from_numpy(mask_eroded).to(device)

    # Delete points in the erosion area of mkpts0
    mask_eroded_tensor_new = mask_eroded_tensor > 0
    points_inside_mask = mask_eroded_tensor_new.nonzero().squeeze().to(device)

    indices = []
    for i, elem in enumerate(mkpts0):
        if elem in points_inside_mask:
            indices.append(i)

    indices = torch.tensor(indices).to(device)
    mconf_f = torch.index_select(mconf, 0, indices)
    mkpts0_f = torch.index_select(mkpts0, 0, indices)
    mkpts1_f = torch.index_select(mkpts1, 0, indices)
    descriptors_img0_f = torch.index_select(descriptors_img0, 0, indices)
    descriptors_img1_f = torch.index_select(descriptors_img1, 0, indices)

    # text = ['LoFTR', 'Matches: {}'.format(len(mkpts0))]
    # img0_raw_np = img0_raw.cpu().detach().numpy()
    # img1_raw_np = img1_raw.cpu().detach().numpy()
    mconf_f_np = mconf_f.cpu().detach().numpy()
    mkpts0_f_np = mkpts0_f.cpu().detach().numpy()
    mkpts1_f_np = mkpts1_f.cpu().detach().numpy()
    descriptors_img0_np = descriptors_img0.cpu().detach().numpy()
    descriptors_img1_np = descriptors_img1.cpu().detach().numpy()
    feature_space0_np = feature_space0.cpu().detach().numpy()
    feature_space1_np = feature_space1.cpu().detach().numpy()

    # Draw
    # color_f_np = cm.jet(mconf_f_np, alpha=0.7)
    #
    # fig = make_matching_figure(img0_raw_np, img1_raw_np, mkpts0_f_np, mkpts1_f_np, color_f_np, mkpts0_f_np, mkpts1_f_np, text)
    # fig.tight_layout(pad=0)
    #
    # fig.canvas.draw()
    # image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # image = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    tuple_for_queue = (
    mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1,
    index_iteration, list_frames, seq_is_end)
    return tuple_for_queue


def LoFTR_pair_relocalization(image_pair, mask, matcher):
    img0_raw = image_pair[0]
    img1_raw = image_pair[1]

    img0_raw = cv2.resize(img0_raw, (NEW_SHAPE, NEW_SHAPE))
    img1_raw = cv2.resize(img1_raw, (NEW_SHAPE, NEW_SHAPE))
    mask[mask != 0] = 255.
    img0_raw = cv2.bitwise_and(img0_raw, mask)
    img1_raw = cv2.bitwise_and(img1_raw, mask)

    img0 = torch.from_numpy(img0_raw)[None][None].to(device) / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].to(device) / 255.
    img0 = kornia.filters.gaussian_blur2d(img0, (9, 9), (1.5, 1.5))
    img1 = kornia.filters.gaussian_blur2d(img1, (9, 9), (1.5, 1.5))
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction all in pytorch
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f']
        mkpts1 = batch['mkpts1_f']
        mconf = batch['mconf']
        descriptors_img0_tensor = matcher.des0
        descriptors_img1_tensor = matcher.des1
        # feature space 3136x256
        feature_space0 = matcher.feat_c0
        feature_space1 = matcher.feat_c1

    descriptors_img0 = descriptors_img0_tensor
    descriptors_img1 = descriptors_img1_tensor

    img0_raw = img0.squeeze()
    img1_raw = img1.squeeze()
    # img1_raw = img1_raw.cpu().detach().numpy()

    # Delete points with a low confidence
    mconf_new = mconf < CONFIDENCE_THRESHOLD
    indices = mconf_new.nonzero().squeeze().to(device)
    mconf = torch.index_select(mconf, 0, indices)
    mkpts0 = torch.index_select(mkpts0, 0, indices)
    mkpts1 = torch.index_select(mkpts1, 0, indices)
    descriptors_img0 = torch.index_select(descriptors_img0, 0, indices)
    descriptors_img1 = torch.index_select(descriptors_img1, 0, indices)

    # Calculation of eroded mask
    erosion_size = 10
    erosion_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    mask_eroded = cv2.erode(mask, element)

    mask_eroded_tensor = torch.from_numpy(mask_eroded).to(device)

    # Delete points in the erosion area of mkpts0
    mask_eroded_tensor_new = mask_eroded_tensor > 0
    points_inside_mask = mask_eroded_tensor_new.nonzero().squeeze().to(device)

    indices = []
    for i, elem in enumerate(mkpts0):
        if elem in points_inside_mask:
            indices.append(i)

    indices = torch.tensor(indices).to(device)
    mconf_f = torch.index_select(mconf, 0, indices)
    mkpts0_f = torch.index_select(mkpts0, 0, indices)
    mkpts1_f = torch.index_select(mkpts1, 0, indices)
    descriptors_img0_f = torch.index_select(descriptors_img0, 0, indices)
    descriptors_img1_f = torch.index_select(descriptors_img1, 0, indices)
    feature_space0 = feature_space0.cpu().detach().numpy()
    feature_space1 = feature_space1.cpu().detach().numpy()

    # Draw
    # color_f_np = cm.jet(mconf_f_np, alpha=0.7)
    #
    # fig = make_matching_figure(img0_raw_np, img1_raw_np, mkpts0_f_np, mkpts1_f_np, color_f_np, mkpts0_f_np, mkpts1_f_np, text)
    # fig.tight_layout(pad=0)
    #
    # fig.canvas.draw()
    # image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # image = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1


def refine_key_frames(list_key_frames):
    #print('Key frames before refinement: ', len(list_key_frames))
    list_dist = []
    print(len(list_key_frames))
    list_indexes_remove = []
    for i in range(len(list_key_frames) - 1):
        for j in range(i + 1, len(list_key_frames) - 1, 1):
            dist = np.linalg.norm(list_key_frames[i][2] - list_key_frames[j][2])
            # dist = torch.cdist(LIST_KEY_FRAMES[i][2],LIST_KEY_FRAMES[j][2],p=2)
            # print('Coppie uguali')
            # print(list_key_frames[i][0])
            # print(list_key_frames[j][0])
            # print(dist)
            list_dist.append(dist)
            if dist < 1300:
                list_indexes_remove.append(list_key_frames[i][0])

    for element in list_key_frames:
        if (element[0] in list_indexes_remove):
            list_key_frames.remove(element)
    print('Key frames after refinement: ', len(list_key_frames))
    return list_key_frames


def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.
    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    if nn_thresh < 0.0:
        raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    # Scaling
    dmat = (dmat - np.min(dmat))/(np.max(dmat) -np.min(dmat))*(1-(-1)) + (-1)

    #Normalization: values smaller than -1 become -1 and values greater than 1 become 1
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    # Get NN indices and scores.

    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]

    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches


# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask, name_experiment, out, initial_panorama):
    path_panorama_image_folder = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment)
    if not os.path.exists(path_panorama_image_folder):
        os.makedirs(path_panorama_image_folder)

    path_panorama_video_folder = os.path.join(DATASET_FILES_PATH, 'output_panorama_video', name_experiment)
    if not os.path.exists(path_panorama_video_folder):
        os.makedirs(path_panorama_video_folder)


    panorama = np.copy(initial_panorama)
    panorama_mask = np.copy(initial_panorama)

    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


    for name in list_images:
        if ('.DS' in name):
            list_images.remove(name)

    print('panorama creation')
    for index, image_name in enumerate(list_images):
        # print(os.path.join(os.getcwd(),path_images,image_name+'.png'))
        image = cv2.imread(os.path.join(os.getcwd(), path_images, image_name), 1)
        image = cv2.resize(image, (NEW_SHAPE, NEW_SHAPE))
        # image_circle = cv2.bitwise_and(image, mask*255)
        image_circle = cv2.bitwise_and(image, mask)
        erosion_size = 30
        erosion_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        mask_eroded = cv2.erode(mask, element)

        canvas_center = np.array(
            [[np.cos(0), np.sin(0), CANVAS_SHAPE / 2], [-np.sin(0), np.cos(0), CANVAS_SHAPE / 2],
             [0, 0, 1]])

        if (index == 0):
            img_origin = np.array([[np.cos(0), np.sin(0), -image_circle.shape[1] / 2],
                                   [-np.sin(0), np.cos(0), -image_circle.shape[0] / 2], [0, 0, 1]])
            H4p = img_origin @ canvas_center


        else:
            # print(index-1)
            H4p_hat = list_matrices[index - 1]
            H4p = H4p_prev @ H4p_hat
            # print(H4p)

        panorama_curr = cv2.warpPerspective(image_circle, np.float32(H4p), (panorama.shape[1], panorama.shape[0]),
                                            flags=cv2.INTER_NEAREST)

        mask_copy = cv2.warpPerspective(mask_eroded, np.float32(H4p), (panorama.shape[1], panorama.shape[0]),
                                        flags=cv2.INTER_NEAREST)

        merge_mertens = cv2.createMergeMertens()
        intersection_mask = cv2.bitwise_and(panorama_mask, mask_copy)
        mask_inv = 255 - intersection_mask;
        intersection_panorama = mask_inv * panorama
        intersection_panorama_curr = mask_inv * panorama_curr

        np.copyto(panorama, panorama_curr, where=mask_copy.astype(bool))
        np.copyto(panorama_mask, mask_copy, where=mask_copy.astype(bool))


        H4p_prev = H4p
        out.write(panorama)
        panorama_mask_gray = cv2.cvtColor(mask_copy, cv2.COLOR_RGB2GRAY)

        transparent = np.zeros((panorama.shape[0], panorama.shape[1], 4), dtype=np.uint8)
        transparent[:, :, 0:3] = panorama_curr
        transparent[:, :, 3] = panorama_mask_gray


        cv2.imwrite(os.path.join(path_image_transparent_folder, image_name), transparent)

    return panorama, out



# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction_second(list_images, list_matrices, path_images, name_video, mask, name_experiment, out, initial_panorama):
    path_panorama_image_folder = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment)
    if not os.path.exists(path_panorama_image_folder):
        os.makedirs(path_panorama_image_folder)

    path_panorama_video_folder = os.path.join(DATASET_FILES_PATH, 'output_panorama_video', name_experiment)
    if not os.path.exists(path_panorama_video_folder):
        os.makedirs(path_panorama_video_folder)


    panorama = np.copy(initial_panorama)
    panorama_mask = np.copy(initial_panorama)

    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


    for name in list_images:
        if ('.DS' in name):
            list_images.remove(name)

    print('panorama creation')

    H4p_prev = INITIAL_MATRIX

    for index, image_name in enumerate(list_images):
        # print(os.path.join(os.getcwd(),path_images,image_name+'.png'))
        image = cv2.imread(os.path.join(os.getcwd(), path_images, image_name), 1)
        image = cv2.resize(image, (NEW_SHAPE, NEW_SHAPE))
        # image_circle = cv2.bitwise_and(image, mask*255)
        image_circle = cv2.bitwise_and(image, mask)
        erosion_size = 30
        erosion_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
        mask_eroded = cv2.erode(mask, element)



        # print(index-1)
        H4p_hat = list_matrices[index]
        H4p = H4p_prev @ H4p_hat

        panorama_curr = cv2.warpPerspective(image_circle, np.float32(H4p), (panorama.shape[1], panorama.shape[0]),
                                            flags=cv2.INTER_NEAREST)
        plt.imshow(panorama_curr)
        plt.show()
        cv2.imwrite('panorama_curr.png', panorama_curr)

        mask_copy = cv2.warpPerspective(mask_eroded, np.float32(H4p), (panorama.shape[1], panorama.shape[0]),
                                        flags=cv2.INTER_NEAREST)


        np.copyto(panorama, panorama_curr, where=mask_copy.astype(bool))
        np.copyto(panorama_mask, mask_copy, where=mask_copy.astype(bool))

        if (index==0):
            edges = cv2.Canny(mask_copy, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) #edges with 3 channels
            dilatation_size = 3
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                               (dilatation_size, dilatation_size))
            dilatation = cv2.dilate(edges, element)
            panorama_with_border = np.copy(panorama)
            np.copyto(panorama_with_border, dilatation, where=dilatation.astype(bool))

            #cv2.imwrite(os.path.join(path_image_transparent_folder, image_name), transparent)
            out.write(panorama_with_border)
            # out.write(panorama)
            panorama_mask_gray = cv2.cvtColor(mask_copy, cv2.COLOR_RGB2GRAY)

            transparent = np.zeros((panorama.shape[0], panorama.shape[1], 4), dtype=np.uint8)
            transparent[:, :, 0:3] = panorama_with_border
            transparent[:, :, 3] = panorama_mask_gray
            #
            cv2.imwrite(os.path.join(path_image_transparent_folder, image_name), transparent)
        else:
            out.write(panorama)
            # out.write(panorama)
            panorama_mask_gray = cv2.cvtColor(mask_copy, cv2.COLOR_RGB2GRAY)

            transparent = np.zeros((panorama.shape[0], panorama.shape[1], 4), dtype=np.uint8)
            transparent[:, :, 0:3] = panorama_curr
            transparent[:, :, 3] = panorama_mask_gray
            #
            cv2.imwrite(os.path.join(path_image_transparent_folder, image_name), transparent)



        H4p_prev = H4p
        # out.write(np.uint8(res_mertens*255))

    return panorama, out


def check_key_frame(queue, lock, list_homographies, list_key_frames):
    # Synchronize access to the console
    SEQ_IS_END = False

    with lock:
        print('Starting consumer => {}'.format(os.getpid()))

    # # Run indefinitely
    loop = 0
    while SEQ_IS_END == False:

        loop = loop + 1
        lock.acquire()
        try:
            # print('sono nel try')
            dictionary_from_queue = queue.get(timeout=1)  # che un po di english ci piace
            # print(type(dictionary_from_queue))

            i = dictionary_from_queue[0]
            list_frames = dictionary_from_queue[1]
            seq_is_end = dictionary_from_queue[2]
            mkpts0_f = dictionary_from_queue[3]
            mkpts1_f = dictionary_from_queue[4]
            mconf_f = dictionary_from_queue[5]
            descriptors_img0 = dictionary_from_queue[6]
            descriptors_img1 = dictionary_from_queue[7]
            feature_space0 = dictionary_from_queue[8]
            feature_space1 = dictionary_from_queue[9]
            mkpts0_f_int = torch.round(mkpts0_f)
            mkpts1_f_int = torch.round(mkpts1_f)
            SEQ_IS_END = seq_is_end
            # Conversion of points to opencv keypoints
            mkpts0_f_np = mkpts0_f.cpu().detach().numpy()
            mkpts1_f_np = mkpts1_f.cpu().detach().numpy()
            mconf_f_np = mconf_f.cpu().detach().numpy()
            h_matrix, inliers = cv2.estimateAffine2D(mkpts1_f_np, mkpts0_f_np)
            row_to_concat = np.array([[0, 0, 1]])
            h_matrix = np.concatenate((h_matrix, row_to_concat), axis=0)
            list_homographies.append(h_matrix)

            # keypoints_img0 = convert_LoFTR_points_to_keypoints(mkpts0_f, mconf_f)
            # keypoints_img1 = convert_LoFTR_points_to_keypoints(mkpts1_f, mconf_f)

            if (i == 0):
                actual_key_frame = list_frames[i]
                H = np.eye(3)
                H_key_frame = INITIAL_MATRIX @ H
                tuple_key_frame = (actual_key_frame, H_key_frame, feature_space0.cpu().numpy(),
                                   descriptors_img0.cpu().numpy())
                tuple_key_frame_copy = deepcopy(tuple_key_frame)
                list_key_frames.append(tuple_key_frame_copy)
                actual_matches_points = descriptors_img1.cpu().numpy()
                points_actual_kf = descriptors_img1.cpu().numpy().shape

            else:

                # Find actual matches: matching from actual_matches_des and current frame descriptors
                # pydevd_pycharm.settrace('10.79.251.35', port=8200, stdoutToServer=True, stderrToServer=True)
                #print(actual_matches_points.shape)
                #print(descriptors_img0.shape)
                matches = nn_match_two_way(np.transpose(actual_matches_points), np.transpose(descriptors_img0.cpu().numpy()), DIS_THRESHOLD)

                actual_matches_points = actual_matches_points[matches[0, :].astype(int)]
                #print('Number of matches points after knn matching : ', len(actual_matches_points))

                #print(points_actual_kf)
                #print(PERCENTAGE*points_actual_kf[0])
                if (actual_matches_points.shape[0] < PERCENTAGE*points_actual_kf[0]):
                    #print('New key frame found')
                    actual_key_frame = list_frames[i]

                    H_key_frame = INITIAL_MATRIX
                    for index in range(i):
                        H_key_frame = H_key_frame @ list_homographies[index - 1]

                    tuple_key_frame = (actual_key_frame, H_key_frame, feature_space0.cpu().numpy(),
                                       descriptors_img0.cpu().numpy())
                    tuple_key_frame_copy = deepcopy(tuple_key_frame)
                    list_key_frames.append(tuple_key_frame_copy)
                    actual_matches_points = descriptors_img1.cpu().numpy()
                    points_actual_kf = descriptors_img1.cpu().numpy().shape
        finally:
            lock.release()


def compute_key_frames_mosaic(list_frames, mask, matcher, name_video, first_matrix):
    list_homographies = []
    list_homographies.append(first_matrix)
    list_key_frames = []
    seq_isend = False

    mask = np.uint8(mask)
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Create the Queue object and the consumer
    queue = mp.Queue()
    lock = mp.Lock()
    # c = mp.spawn(check_key_frame, args=(queue, lock), nprocs=1, join=True, daemon=True, start_method='spawn')
    with Manager() as manager:

        list_homographies = manager.list(list_homographies)
        list_key_frames = manager.list(list_key_frames)

        c = mp.Process(target=check_key_frame, args=(queue, lock, list_homographies, list_key_frames))
        c.daemon = True
        c.start()

        t = trange(len(list_frames) - 1)
        # Key frame extraction
        for i in t:
            # print(i)
            start = datetime.now()
            if (i == len(t) - 1):
                seq_isend = True
            img0_pth = os.path.join(DATASET_PATH, name_video, 'images', list_frames[i])
            img1_pth = os.path.join(DATASET_PATH, name_video, 'images',
                                    list_frames[i + 1])
            img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
            image_pair = [img0_raw, img1_raw]
            tuple_loftr = LoFTR_pair(image_pair, mask, matcher, i, list_frames, seq_isend)
            dict_from_tuple = [
                tuple_loftr[7], tuple_loftr[8], tuple_loftr[9], tuple_loftr[0],
                tuple_loftr[1], tuple_loftr[2], tuple_loftr[3],
                tuple_loftr[4], tuple_loftr[5], tuple_loftr[6]
            ]
            queue.put(dict_from_tuple)
            t.set_postfix(time=datetime.now() - start)

        list_homographies_final = list(list_homographies)
        list_key_frames_final = deepcopy(list_key_frames)
        list_key_frames_final = list(list_key_frames_final)
        c.join()
        return list_homographies_final, list_key_frames_final



def relocalize(frame_to_relocalize, matcher, mask, list_key_frames, path_frames):

    list_dist = []
    list_key_frames_images = []
    list_matrices_reattach = []

    for elem in list_key_frames:
        path_key_frame = path_frames + '/' + elem[0]
        key_frame = cv2.imread(path_key_frame, 1)
        key_frame = cv2.resize(key_frame, (NEW_SHAPE, NEW_SHAPE))
        list_key_frames_images.append(key_frame)


    frame_to_relocalize = cv2.cvtColor(frame_to_relocalize, cv2.COLOR_RGB2GRAY)
    count = 0
    for key_frame in list_key_frames_images:
        count = count + 1
        key_frame = cv2.cvtColor(key_frame, cv2.COLOR_RGB2GRAY)
        image_pair = [frame_to_relocalize, key_frame]
        mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1 = LoFTR_pair_relocalization(image_pair, mask, matcher)
        mkpts0_f_np = mkpts0_f.cpu().numpy()
        mkpts1_f_np = mkpts1_f.cpu().numpy()
        dist = np.linalg.norm(feature_space0 - feature_space1)
        list_dist.append(dist)
        h_matrix, inliers = cv2.estimateAffine2D(mkpts1_f_np, mkpts0_f_np)
        row_to_concat = np.array([[0, 0, 1]])
        h_matrix = np.concatenate((h_matrix, row_to_concat), axis=0)
        list_matrices_reattach.append(h_matrix)

    min_dist = min(list_dist)
    min_index = list_dist.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]
    min_matrix = list_matrices_reattach[min_index]
    return min_index, key_frame_min_dist, min_matrix





# MAIN FLOW
def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(os.path.join(os.getcwd(), "code_Git_imported/LoFTR/weights/outdoor_ds.ckpt"))['state_dict'])
    matcher = matcher.eval().to(device)

    name_video = 'anon016_CLIP18'
    path_frames = os.path.join(DATASET_PATH, name_video, 'images')
    mask = cv2.imread(os.path.join(DATASET_PATH, name_video, 'mask.png'), 0)
    mask = cv2.resize(mask, (NEW_SHAPE, NEW_SHAPE))

    list_frames = sorted(os.listdir(path_frames))  # list containing all the images names

    path_frame_to_relocalize = os.path.join(path_frames, 'Video016_CLIP18_00095.png')
    frame_to_relocalize = 'Video016_CLIP18_00095.png'

    list_first_part = []
    i=0
    while (list_frames[i] != frame_to_relocalize):
        list_first_part.append(list_frames[i])
        i=i+1

    list_second_part = []
    for index in range(i, len(list_frames),1):
        list_second_part.append(list_frames[index])


    list_homographies_first, list_key_frames_first = compute_key_frames_mosaic(list_first_part, mask, matcher, name_video, np.eye(3))

    list_key_frames_first = refine_key_frames(list_key_frames_first)

    # parameters for the video
    size = (CANVAS_SHAPE, CANVAS_SHAPE)
    fps = 15
    path_video = os.path.join(DATASET_FILES_PATH, 'output_panorama_video', name_experiment,
                              name_video + '.mp4')
    out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

    initial_panorama= np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
    panorama_first, out = panorama_reconstruction(list_first_part, list_homographies_first, path_frames, name_video, mask, name_experiment, out, initial_panorama)

    frame_to_relocalize_img = cv2.imread(path_frame_to_relocalize, 1)
    frame_to_relocalize_img = cv2.resize(frame_to_relocalize_img, (NEW_SHAPE, NEW_SHAPE))

    min_index, key_frame_min_dist, min_matrix = relocalize(frame_to_relocalize_img, matcher, mask, list_key_frames_first, path_frames)
    global INITIAL_MATRIX
    INITIAL_MATRIX = key_frame_min_dist[1]

    list_homographies_second, list_key_frames_second = compute_key_frames_mosaic(list_second_part, mask, matcher, name_video, min_matrix)
    panorama_second, out = panorama_reconstruction_second(list_second_part, list_homographies_second, path_frames,
                                                        name_video, mask, name_experiment, out, panorama_first)

    out.release()
    path_panorama = os.path.join(DATASET_FILES_PATH, 'output_panorama_images', name_experiment,
                                 'panorama_' + name_video + '.png')
    # print(path_panorama)
    cv2.imwrite(path_panorama, panorama_second)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

