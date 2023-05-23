import os
from copy import deepcopy
import imutils
from skimage import measure
from tqdm import trange
import kornia
import pandas as pd
import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
# np.set_printoptions(threshold=sys.maxsize)
#import pydevd_pycharm
from datetime import datetime
from code_Git_imported.LoFTR.src.loftr import LoFTR, default_cfg
#from exposure_fusion.exposure_fusion import load_images_list, exposure_fusion, load_images

# Global variables
NEW_SHAPE = 448
DIS_THRESHOLD = 100
CONFIDENCE_THRESHOLD = 0.75
PERCENTAGE = 0.10
CANVAS_SHAPE = 4000
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
name_experiment = 'SLAM_LoFTR_NO_PARALLEL'  # string containing the name used to identify the experiment
DATASET_PATH = os.path.join(os.getcwd(), '../final_dataset')
DATASET_FILES_PATH = os.path.join(os.getcwd(), '../final_dataset_files')
if not os.path.exists(DATASET_FILES_PATH):
    os.makedirs(DATASET_FILES_PATH)



canvas_center = np.array(
    [[np.cos(0), np.sin(0), CANVAS_SHAPE / 2], [-np.sin(0), np.cos(0), CANVAS_SHAPE / 2], [0, 0, 1]])
img_origin = np.array([[np.cos(0), np.sin(0), -NEW_SHAPE / 2], [-np.sin(0), np.cos(0), -NEW_SHAPE / 2], [0, 0, 1]])
INITIAL_MATRIX = img_origin @ canvas_center


def build_video_key_frame(list_key_frames, name_video, path_frames, mask, list_homographies, list_frames):
    black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
    panorama = np.copy(black_canvas)
    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    key_frames_names = [x[0] for x in list_key_frames]

    # parameters for the video
    size = (black_canvas.shape[1], black_canvas.shape[0])
    fps = 25
    path_video = os.path.join(DATASET_FILES_PATH,'output_panorama_video', name_experiment,
                              name_video + '_key_frames.mp4')
    out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

    for index, frame in enumerate(list_frames):
        if frame in key_frames_names:
            index_kf = [y[0] for y in list_key_frames].index(frame)
            erosion_size = 10
            erosion_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
            mask_eroded = cv2.erode(mask, element)
            # panorama_curr = cv2.warpPerspective(image_circle, np.float32(key_frame[1]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(list_key_frames[index_kf][1]),
                                            (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
            regions = measure.regionprops(mask_copy)
            for props in regions:
                centroid = (np.array(props.centroid)).astype(int)
            edges = cv2.Canny(mask_copy, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
            dilatation_size = 1
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                                (dilatation_size, dilatation_size))
            dilatation = cv2.dilate(edges, element)
            b, g, r = cv2.split(dilatation)
            b[b == 255] = 0
            g[g == 255] = 0
            dilatation = cv2.merge([b, g, r])  # dilatation with red color
            np.copyto(panorama, dilatation, where=dilatation.astype(bool))
            cv2.circle(panorama, (centroid[1], centroid[0]), 3, (0, 255, 0), -1)

            out.write(panorama)


        else:

            if (index == 0):
                H = INITIAL_MATRIX
            else:
                H = INITIAL_MATRIX
                for i in range(index - 1):
                    H = H @ list_homographies[i]

            erosion_size = 10
            erosion_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
            mask_eroded = cv2.erode(mask, element)
            # panorama_curr = cv2.warpPerspective(image_circle, np.float32(key_frame[1]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(H), (panorama.shape[1], panorama.shape[0]),
                                            flags=cv2.INTER_NEAREST)
            regions = measure.regionprops(mask_copy)
            for props in regions:
                centroid = (np.array(props.centroid)).astype(int)
            edges = cv2.Canny(mask_copy, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels

            dilatation_size = 1
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                                (dilatation_size, dilatation_size))
            dilatation = cv2.dilate(edges, element)
            b, g, r = cv2.split(dilatation)
            r[r == 255] = 0
            g[g == 255] = 0
            dilatation = cv2.merge([b, g, r])  # dilatation with red color
            panorama_curr = np.copy(panorama)
            np.copyto(panorama_curr, dilatation, where=dilatation.astype(bool))
            cv2.circle(panorama_curr, (centroid[1], centroid[0]), 3, (255, 0, 0), -1)
            # plt.imshow(panorama_curr)
            # plt.show()
            out.write(panorama_curr)

    out.release()
    path_panorama = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment,
                                 'panorama_' + name_video + '_key_frames_after.png')
    # print(path_panorama)
    cv2.imwrite(path_panorama, panorama)
    cv2.destroyAllWindows()
    print('key frames video done!')


def build_video_key_frame_before(list_key_frames, name_video, path_frames, mask, list_homographies, list_frames):
    black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
    panorama = np.copy(black_canvas)
    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    key_frames_names = [x[0] for x in list_key_frames]

    # parameters for the video
    size = (black_canvas.shape[1], black_canvas.shape[0])
    fps = 25
    path_video = os.path.join(DATASET_FILES_PATH,'output_panorama_video', name_experiment,
                              name_video + '_key_frames_before.mp4')
    out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

    for index, frame in enumerate(list_frames):
        if frame in key_frames_names:
            index_kf = [y[0] for y in list_key_frames].index(frame)
            erosion_size = 10
            erosion_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
            mask_eroded = cv2.erode(mask, element)
            # panorama_curr = cv2.warpPerspective(image_circle, np.float32(key_frame[1]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(list_key_frames[index_kf][1]),
                                            (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
            regions = measure.regionprops(mask_copy)
            for props in regions:
                centroid = (np.array(props.centroid)).astype(int)
            edges = cv2.Canny(mask_copy, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
            dilatation_size = 1
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                                (dilatation_size, dilatation_size))
            dilatation = cv2.dilate(edges, element)
            b, g, r = cv2.split(dilatation)
            b[b == 255] = 0
            g[g == 255] = 0
            dilatation = cv2.merge([b, g, r])  # dilatation with red color
            np.copyto(panorama, dilatation, where=dilatation.astype(bool))
            cv2.circle(panorama, (centroid[1], centroid[0]), 3, (0, 255, 0), -1)

            out.write(panorama)


        else:

            if (index == 0):
                H = INITIAL_MATRIX
            else:
                H = INITIAL_MATRIX
                for i in range(index - 1):
                    H = H @ list_homographies[i]

            erosion_size = 10
            erosion_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
            mask_eroded = cv2.erode(mask, element)
            # panorama_curr = cv2.warpPerspective(image_circle, np.float32(key_frame[1]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(H), (panorama.shape[1], panorama.shape[0]),
                                            flags=cv2.INTER_NEAREST)
            regions = measure.regionprops(mask_copy)
            for props in regions:
                centroid = (np.array(props.centroid)).astype(int)
            edges = cv2.Canny(mask_copy, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels

            dilatation_size = 1
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                                (dilatation_size, dilatation_size))
            dilatation = cv2.dilate(edges, element)
            b, g, r = cv2.split(dilatation)
            r[r == 255] = 0
            g[g == 255] = 0
            dilatation = cv2.merge([b, g, r])  # dilatation with red color
            panorama_curr = np.copy(panorama)
            np.copyto(panorama_curr, dilatation, where=dilatation.astype(bool))
            cv2.circle(panorama_curr, (centroid[1], centroid[0]), 3, (255, 0, 0), -1)
            # plt.imshow(panorama_curr)
            # plt.show()
            out.write(panorama_curr)

    out.release()
    path_panorama = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment,
                                 'panorama_' + name_video + '_key_frames_before.png')
    # print(path_panorama)
    cv2.imwrite(path_panorama, panorama)
    cv2.destroyAllWindows()
    print('key frames video done!')


def LoFTR_pair(image_pair, mask, matcher, index_iteration, list_frames):
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

    tuple_for_queue = (
    mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1,
    index_iteration, list_frames)
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


    return mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1


def refine_key_frames(list_key_frames):
    print('Key frames before refinement: ', len(list_key_frames))
    list_dist = []
    list_indexes_remove = []
    for i in range(len(list_key_frames) - 1):
        for j in range(i + 1, len(list_key_frames) - 1, 1):
            dist = np.linalg.norm(list_key_frames[i][2] - list_key_frames[j][2])
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
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask, name_experiment):
    path_panorama_image_folder = os.path.join(DATASET_FILES_PATH,'output_panorama_images',
                                              name_experiment)
    if not os.path.exists(path_panorama_image_folder):
        # print('creating new panorama image folder')
        os.makedirs(path_panorama_image_folder)

    path_panorama_video_folder = os.path.join(DATASET_FILES_PATH,'output_panorama_video',
                                              name_experiment)
    if not os.path.exists(path_panorama_video_folder):
        # print('creating new panorama video folder')
        os.makedirs(path_panorama_video_folder)

    path_panorama_image = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment,
                                       'panorama_' + name_video + '.png')
    if not os.path.exists(path_panorama_image):
        black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
        panorama = np.copy(black_canvas)
        panorama_mask = np.copy(black_canvas)

        mask = np.uint8(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # parameters for the video
        size = (black_canvas.shape[1], black_canvas.shape[0])
        fps = 25
        path_video = os.path.join(DATASET_FILES_PATH,'output_panorama_video', name_experiment,
                                  name_video + '.mp4')
        out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

        for name in list_images:
            if ('anon' not in name):
                list_images.remove(name)

        print('panorama creation')
        for index, image_name in enumerate(list_images):
            # print(os.path.join(os.getcwd(),path_images,image_name+'.png'))
            image = cv2.imread(os.path.join(os.getcwd(), path_images, image_name), 1)
            image = cv2.resize(image, (NEW_SHAPE, NEW_SHAPE))
            # image_circle = cv2.bitwise_and(image, mask*255)
            image_circle = cv2.bitwise_and(image, mask)
            erosion_size = 10
            erosion_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
            mask_eroded = cv2.erode(mask, element)

            canvas_center = np.array(
                [[np.cos(0), np.sin(0), black_canvas.shape[1] / 2], [-np.sin(0), np.cos(0), black_canvas.shape[0] / 2],
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


            np.copyto(panorama, panorama_curr, where=mask_copy.astype(bool))
            np.copyto(panorama_mask, mask_copy, where=mask_copy.astype(bool))



            H4p_prev = H4p
            out.write(panorama)

        out.release()
        path_panorama = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment,
                                     'panorama_' + name_video + '.png')
        # print(path_panorama)
        cv2.imwrite(path_panorama, panorama)
        cv2.destroyAllWindows()

        return panorama

    else:
        print('Panorama already reconstructed')
        panorama = cv2.imread(path_panorama_image, 1)
        return panorama


def panorama_metric_computation(list_frames, list_homographies, path_frames, name_video):

    path_boxplot = os.path.join(DATASET_FILES_PATH,'boxplots_npy', name_experiment)
    if not os.path.exists(path_boxplot):
        # print('creating new panorama image folder')
        os.makedirs(path_boxplot)


    list_ssim = []
    for i in range(0,len(list_frames)-5,1):
        img1 = list_frames[i]
        img2 = list_frames[i+5]

        matrix1_2 = np.eye(3)
        for index in range(i, i+5, 1):
            matrix1_2 = matrix1_2 @list_homographies[index]
        ssim_index = compute_metric(img1, img2, matrix1_2, path_frames)

        for j in range(i, i + 5 - 1, 1):
            diff = np.abs(np.subtract(list_homographies[j + 1][0:1,0:1], list_homographies[j + 1][0:1,0:1]))
            diff_t = np.abs(np.subtract(list_homographies[j + 1][0:1,2], list_homographies[j + 1][0:1,2]))
            if (diff.any()>0.4) or (diff_t>60):
                ssim_index = 0

        list_ssim.append(ssim_index)

    np_ssim = np.array(list_ssim)
    np.save(os.path.join(path_boxplot,'ssim_mosaicking_'+name_video), np_ssim)



def check_key_frame(i, list_frames, mkpts0_f, mkpts1_f, mconf_f, descriptors_img0, descriptors_img1,feature_space0, feature_space_1, actual_matches_points, points_actual_kf, list_homographies, list_key_frames):


        # Conversion of points to opencv keypoints
        mkpts0_f_np = mkpts0_f.cpu().detach().numpy()
        mkpts1_f_np = mkpts1_f.cpu().detach().numpy()
        mconf_f_np = mconf_f.cpu().detach().numpy()
        h_matrix, inliers = cv2.estimateAffine2D(mkpts1_f_np, mkpts0_f_np)
        row_to_concat = np.array([[0, 0, 1]])
        h_matrix = np.concatenate((h_matrix, row_to_concat), axis=0)
        list_homographies.append(h_matrix)


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
            matches = nn_match_two_way(np.transpose(actual_matches_points), np.transpose(descriptors_img0.cpu().numpy()), DIS_THRESHOLD)

            actual_matches_points = actual_matches_points[matches[0, :].astype(int)]

            if (actual_matches_points.shape[0] < PERCENTAGE*points_actual_kf[0]):
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

        return list_homographies, list_key_frames, actual_matches_points, points_actual_kf


def compute_key_frames_mosaic(list_frames, mask, matcher, name_video):
    list_homographies = []
    list_homographies.append(np.eye(3))
    list_key_frames = []
    actual_matches_points = []
    points_actual_kf = []


    mask = np.uint8(mask)

    t = trange(len(list_frames) - 1)
    # Key frame extraction
    for i in t:
        print(i)
        start = datetime.now()
        if (i == len(t) - 1):
            seq_isend = True
        img0_pth = os.path.join(DATASET_PATH, name_video, 'images', list_frames[i])
        img1_pth = os.path.join(DATASET_PATH, name_video, 'images', list_frames[i + 1])
        img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
        image_pair = [img0_raw, img1_raw]
        tuple_loftr = LoFTR_pair(image_pair, mask, matcher, i, list_frames)

        list_homographies, list_key_frames, actual_matches_points, points_actual_kf = check_key_frame(i, list_frames, tuple_loftr[0], tuple_loftr[1], tuple_loftr[2], tuple_loftr[3],
            tuple_loftr[4], tuple_loftr[5], tuple_loftr[6], actual_matches_points, points_actual_kf, list_homographies, list_key_frames)
        t.set_postfix(time=datetime.now() - start)


    return list_homographies, list_key_frames


# It adds a transformation to an image to perform sanity checks and returns the transformed image
# with a string indicating the applied transformation
# Possible transformation: rotation, illumination change (contrast and brightness change), scaling (through crop reduction),
# and addiction of a black patch to simulate an occlusion in the centre of the image
def add_transformation(image, ill_alpha, ill_beta, rot, crop_red, patch_dim):
    # line to apply rotation
    image = imutils.rotate(image, rot)

    # line to apply illumination changes
    image = cv2.convertScaleAbs(image, alpha=ill_alpha, beta=ill_beta)

    # lines to apply scaling
    shape = image.shape[0]
    image = image[crop_red:image.shape[1] - crop_red, crop_red:image.shape[0] - crop_red]
    image = cv2.resize(image, (shape, shape))

    # lines to add a black patch in the centre
    patch = np.zeros((patch_dim, patch_dim, 3), dtype='uint8')
    x_offset = (image.shape[0] - patch.shape[0]) // 2
    y_offset = (image.shape[1] - patch.shape[1]) // 2
    image[y_offset:y_offset + patch.shape[1], x_offset:x_offset + patch.shape[0]] = patch
    return image


def relocalize(frame_to_relocalize_transf, matcher, mask, list_key_frames, list_key_frames_names, name_frame_to_relocalize):
    list_dist = []
    list_matrices_reattach = []
    frame_to_relocalize_transf = cv2.cvtColor(frame_to_relocalize_transf, cv2.COLOR_RGB2GRAY)
    count = 0
    for key_frame in list_key_frames:
        count = count + 1
        key_frame = cv2.cvtColor(key_frame, cv2.COLOR_RGB2GRAY)
        image_pair = [frame_to_relocalize_transf, key_frame]
        mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1 = LoFTR_pair_relocalization(
            image_pair, mask, matcher)
        mkpts0_f_np = mkpts0_f.cpu().numpy()
        mkpts1_f_np = mkpts1_f.cpu().numpy()
        dist = np.linalg.norm(feature_space0 - feature_space1)
        print(dist)
        list_dist.append(dist)
        h_matrix, inliers = cv2.estimateAffine2D(mkpts1_f_np, mkpts0_f_np)
        row_to_concat = np.array([[0, 0, 1]])
        h_matrix = np.concatenate((h_matrix, row_to_concat), axis=0)
        list_matrices_reattach.append(h_matrix)


    min_dist = min(list_dist)
    min_index = list_dist.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]
    key_frame_min_dist_name = list_key_frames_names[min_index][0]
    min_matrix = list_matrices_reattach[min_index]
    tuple_result = (min_index, key_frame_min_dist, key_frame_min_dist_name, min_matrix)
    print('frame to relocalize', name_frame_to_relocalize)
    print('key frame min dist', key_frame_min_dist_name)
    print('min dist', min_dist)
    return tuple_result


def compute_metric(img1_name, img2_name, mat, path_frames):
    img1 = cv2.imread(os.path.join(path_frames, img1_name), 0)
    img2 = cv2.imread(os.path.join(path_frames, img2_name), 0)

    img2_warped = cv2.warpPerspective(img2, np.float32(mat),(img2.shape[1], img2.shape[0]),flags=cv2.INTER_NEAREST)

    crop = 60/100*img1.shape[0]
    x = img1.shape[1]/2 - crop / 2
    y = img1.shape[0]/2 - crop / 2

    crop_img1 = img1[int(y):int(y + crop), int(x):int(x + crop)]
    crop_img2 = img2_warped[int(y):int(y + crop), int(x):int(x + crop)]


    img1_gray_gb = cv2.GaussianBlur(crop_img1, (9, 9), sigmaX=1.5, sigmaY=1.5, borderType =cv2.BORDER_DEFAULT)
    img2_gray_gb = cv2.GaussianBlur(crop_img2, (9, 9), sigmaX=1.5, sigmaY=1.5, borderType =cv2.BORDER_DEFAULT)
    ssim_index = ssim(img1_gray_gb, img2_gray_gb)
    return ssim_index



# It performs the sanity check  between the frame to relocalized with an applied transformation and the found keyframes
def sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize,
                 string_name_panorama, name_video, list_key_frames):
    list_key_frames_images = []
    panorama = panorama * 0.5

    for i in range(len(list_key_frames)):
        path_key_frame = path_frames + '/' + list_key_frames[i][0]
        key_frame = cv2.imread(path_key_frame, 1)
        key_frame = cv2.resize(key_frame, (NEW_SHAPE, NEW_SHAPE))
        list_key_frames_images.append(key_frame)

    tuple_result = relocalize(frame_to_relocalize_transf, matcher, mask, list_key_frames_images, list_key_frames, name_frame_to_relocalize)

    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # METRIC CALCULATION

    metric_ssim = compute_metric(name_frame_to_relocalize,tuple_result[2], tuple_result[3], path_frames)
    print(metric_ssim)

    # RESULT REPRESENTATION

    key_frame_warped = cv2.warpPerspective(tuple_result[1], np.float32(list_key_frames[tuple_result[0]][1]),
                                           (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
    mask_key_frame = cv2.warpPerspective(mask, np.float32(list_key_frames[tuple_result[0]][1]),
                                         (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
    mask_key_frame_copy = np.copy(mask_key_frame)
    panorama_with_key_frame = np.copy(panorama)
    np.copyto(panorama_with_key_frame, key_frame_warped, where=mask_key_frame.astype(bool))

    edges = cv2.Canny(mask_key_frame_copy, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
    dilatation_size = 10
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation = cv2.dilate(edges, element)
    b, g, r = cv2.split(dilatation)
    b[b == 255] = 0
    g[g == 255] = 0
    dilatation = cv2.merge([b, g, r])  # dilatation with red color
    np.copyto(panorama_with_key_frame, dilatation, where=dilatation.astype(bool))

    image_to_check_warped = cv2.warpPerspective(frame_to_relocalize_transf, np.float32(matrix_frame_to_relocalize),
                                                (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
    mask_image_to_check = cv2.warpPerspective(mask, np.float32(matrix_frame_to_relocalize),
                                              (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
    mask_image_to_check_copy = np.copy(mask_image_to_check)

    np.copyto(panorama_with_key_frame, image_to_check_warped, where=mask_image_to_check.astype(bool))
    edges = cv2.Canny(mask_image_to_check_copy, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
    dilatation_size = 10
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation = cv2.dilate(edges, element)
    b, g, r = cv2.split(dilatation)
    r[r == 255] = 0
    g[g == 255] = 0
    dilatation = cv2.merge([b, g, r])  # dilatation with red color
    np.copyto(panorama_with_key_frame, dilatation, where=dilatation.astype(bool))

    path_folder = os.path.join(DATASET_FILES_PATH,'sanity_check_panorama', name_experiment)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    path_folder = os.path.join(DATASET_FILES_PATH,'sanity_check_panorama', name_experiment,
                               name_video)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    cv2.imwrite(os.path.join(path_folder, string_name_panorama), panorama_with_key_frame)
    tuple_to_return = (tuple_result[0], list_key_frames[tuple_result[0]][0])
    return tuple_to_return, metric_ssim


# MAIN FLOW
def main():

    list_videos = sorted(
        os.listdir(os.path.join(DATASET_PATH)))  # contains list of videos names

    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(os.path.join(os.getcwd(), "../code_Git_imported/LoFTR/weights/outdoor_ds.ckpt"))['state_dict'])
    matcher = matcher.eval().to(device)

    index_column = 0
    index_row = 0

    experiment_result = np.zeros((18, len(list_videos)), dtype=object)

    for name_video in list_videos:
        print(name_video)
        path_frames = os.path.join(DATASET_PATH, name_video, 'images')
        # mask = mask/255. #binary mask
        mask = cv2.imread(os.path.join(DATASET_PATH, name_video, 'mask.png'), 0)
        mask = cv2.resize(mask, (NEW_SHAPE, NEW_SHAPE))

        list_frames = sorted(os.listdir(path_frames))  # list containing all the images names
        list_homographies, list_key_frames = compute_key_frames_mosaic(list_frames, mask, matcher, name_video)
        list_key_frames = refine_key_frames(list_key_frames)

        build_video_key_frame(list_key_frames, name_video, path_frames, mask, list_homographies, list_frames)

        panorama = panorama_reconstruction(list_frames, list_homographies, path_frames, name_video, mask,
                                           name_experiment)
        panorama_metric_computation(list_frames, list_homographies, path_frames, name_video)


        num_frame_to_relocalize = 51
        name_frame_to_relocalize = list_frames[num_frame_to_relocalize]
        if (name_frame_to_relocalize in list_key_frames):
            num_frame_to_relocalize = 50
            name_frame_to_relocalize = list_frames[num_frame_to_relocalize]

        path_frame_to_relocalize = os.path.join(path_frames, name_frame_to_relocalize)
        matrix_frame_to_relocalize = INITIAL_MATRIX

        for i in range(num_frame_to_relocalize):
            matrix_frame_to_relocalize = matrix_frame_to_relocalize @ list_homographies[i]

        frame_to_relocalize = cv2.imread(path_frame_to_relocalize, 1)
        frame_to_relocalize = cv2.resize(frame_to_relocalize, (NEW_SHAPE, NEW_SHAPE))

        # LIST OF ALL THE SANITY CHECKS
        list_ssim_sanity_check = []
        # sanity check with no transformation applied
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[0, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with 10 degrees rotation
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 10
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[1, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with 30 degrees rotation
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 30
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[2, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with 60 degrees rotation
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 60
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[3, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast alpha=0.8
        ill_alpha = 0.80
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[4, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast alpha=0.9
        ill_alpha = 0.90
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[5, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast alpha=1.10
        ill_alpha = 1.10
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[6, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast alpha=1.20
        ill_alpha = 1.20
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[7, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast beta=0.8
        ill_alpha = 1.0
        ill_beta = 0.8
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[8, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast beta=0.9
        ill_alpha = 1.0
        ill_beta = 0.9
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[9, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast beta=1.10
        ill_alpha = 1.0
        ill_beta = 1.10
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[10, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with contrast beta=1.20
        ill_alpha = 1.0
        ill_beta = 1.20
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[11, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check crop reduction=20
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 20
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[12, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check crop reduction=30
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 30
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[13, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check crop reduction=50
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 50
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[14, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        # sanity check with patch insertion of patch_dim=100
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 100
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red,
                                                        patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(
            ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result, metric_ssim = sanity_check(name_frame_to_relocalize, frame_to_relocalize_transf, path_frames, matcher, panorama, mask,
                                    matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[15, index_column] = tuple_result
        list_ssim_sanity_check.append(metric_ssim)

        array_ssim_sanity_check = np.array(list_ssim_sanity_check)
        np.save(os.path.join(os.path.join(DATASET_FILES_PATH,'boxplots_npy', name_experiment), 'ssim_relocalization_' + name_video), array_ssim_sanity_check)

        # code to add the interval of key frames between which we can find the frame to reattach
        list_key_frames_num = [item[0][8:] for item in list_key_frames]
        frame_to_reattach_num = name_frame_to_relocalize[8:]


        list_tuples_frames_num = []
        for i in range(len(list_key_frames_num)):
            list_tuples_frames_num.append((list_key_frames_num[i], i))

        first_frame = list_key_frames_num[0]
        last_frame = list_key_frames_num[0]

        min_interval = 0
        max_interval = len(list_key_frames_num)

        for i in range(len(list_key_frames_num)):
            if frame_to_reattach_num > list_tuples_frames_num[i][0]:
                min_interval = i

        for i in reversed(range(len(list_key_frames))):
            if frame_to_reattach_num < list_tuples_frames_num[i][0]:
                max_interval = i

        interval = (min_interval, max_interval)
        tuple_frame_to_reattach = (frame_to_relocalize, interval)

        experiment_result[16, index_column] = tuple_frame_to_reattach
        res_list = [x[0] for x in list_key_frames]
        experiment_result[17, index_column] = res_list

        index_column = index_column + 1

    # transform array experiment result in an external result sheet

    # convert array to pandas dataframe
    # print(experiment_result)
    df = pd.DataFrame(experiment_result)

    # add titles to the columns
    df.columns = list_videos
    df.index = ['1) No Transf', '2) Rot=10', '3) Rot=30', '4) Rot=60', '5) Contr=0.8',
                '6) Contr=0.9', '7) Contr=1.1', '8) Contr=1.2', '9) Bright=0.8', '10) Bright=0.9',
                '11) Bright=1.1', '12) Bright=1.2', '13) Crop=20', '14) Crop=30', '15) Crop=50',
                '16) Patch=100px', '17) Key-frame to reattach', '18) List of key-frames']

    ## save to xlsx file
    path_folder_experiments = os.path.join(DATASET_FILES_PATH, 'experiments_files')
    if not os.path.exists(path_folder_experiments):
        os.makedirs(path_folder_experiments)

    filepath = os.path.join(path_folder_experiments, name_experiment + '.xlsx')
    df.to_excel(filepath, index=True)


if __name__ == '__main__':
    main()

