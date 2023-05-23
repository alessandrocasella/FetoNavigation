import os
import sys

import imutils
import kornia
import pandas as pd
import torch
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.cm as cm
from datetime import datetime

from code_Git_imported.LoFTR.src.utils.plotting import make_matching_figure
from code_Git_imported.LoFTR.src.loftr import LoFTR, default_cfg




def LoFTR_pair(image_pair, mask, matcher):
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
    #img1_raw = img1_raw.cpu().detach().numpy()

    # Delete points with a low confidence
    mconf_new = mconf<0.60
    indices = mconf_new.nonzero().squeeze().to(device)
    mconf = torch.index_select(mconf, 0, indices)
    mkpts0 = torch.index_select(mkpts0, 0, indices)
    mkpts1 = torch.index_select(mkpts1, 0, indices)
    descriptors_img0 = torch.index_select(descriptors_img0, 0, indices)
    descriptors_img1 = torch.index_select(descriptors_img1, 0, indices)



    # Calculation of eroded mask
    erosion_size = 10
    erosion_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
    mask_eroded = cv2.erode(mask, element)

    mask_eroded_tensor = torch.from_numpy(mask_eroded).to(device)

    # Delete points in the erosion area of mkpts0
    mask_eroded_tensor_new = mask_eroded_tensor>0
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


    text = ['LoFTR', 'Matches: {}'.format(len(mkpts0))]
    img0_raw_np = img0_raw.cpu().detach().numpy()
    img1_raw_np = img1_raw.cpu().detach().numpy()
    mconf_f_np = mconf_f.cpu().detach().numpy()
    mkpts0_f_np = mkpts0_f.cpu().detach().numpy()
    mkpts1_f_np = mkpts1_f.cpu().detach().numpy()


    # Draw
    color_f_np = cm.jet(mconf_f_np, alpha=0.7)

    fig = make_matching_figure(img0_raw_np, img1_raw_np, mkpts0_f_np, mkpts1_f_np, color_f_np, mkpts0_f_np, mkpts1_f_np, text)
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image, mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1

def convert_LoFTR_points_to_keypoints(pts, confidence):
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        for i in range(len(pts)):
            keypoint = cv2.KeyPoint(pts[i][0], pts[i][1], size=1, response=confidence[i])
            kps.append(keypoint)
    return kps

def compute_key_frames (list_frames, mask, matcher, initial_matrix, name_video):
    list_key_frames = []
    list_descriptors_key_frame = []
    list_homographies = []
    matrices_key_frames = []
    list_homographies.append(np.eye(3))
    matrices_key_frames.append(initial_matrix)
    # fps = 25.0
    # path_video = os.path.join(os.getcwd(), 'ouput_LOFTR_anon001.mp4')
    # out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (750, 450), True)

    # Key frame extraction
    for i in range(len(list_frames) - 1):
        print(i)
        img0_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images', list_frames[i])
        img1_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images', list_frames[i + 1])
        img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
        image_pair = [img0_raw, img1_raw]

        image, mkpts0_f, mkpts1_f, mconf_f, descriptors_img0, descriptors_img1, feature_space0, feature_space1 = LoFTR_pair(image_pair, mask, matcher)

        mkpts0_f_int = torch.round(mkpts0_f)
        mkpts1_f_int = torch.round(mkpts1_f)
        # out.write(image)
        # np.savetxt('First_'+str(i)+'.txt', mkpts0_f_int, delimiter=',', fmt='%1.4f')
        # np.savetxt('Second_'+str(i)+'.txt', mkpts1_f_int, delimiter=',', fmt='%1.4f')

        #start = datetime.now()
        if (i == 0):
            actual_key_frame = list_frames[i]
            list_key_frames.append(actual_key_frame)
            actual_matches_points = mkpts1_f_int
            list_descriptors_key_frame.append(descriptors_img0)


        else:
            #print(type(mkpts0_f_int))
            # Find actual matches: keypoints from previous iterations that are present also in this iteration
            #start = datetime.now()
            values, indices = torch.topk(((actual_matches_points.t() == mkpts0_f_int.unsqueeze(-1)).all(dim=1)).int(), 1, 1)
            indices = indices[values != 0]

            # pydevd_pycharm.settrace('localhost', port=8200, stdoutToServer=True, stderrToServer=True)
            actual_matches_points = actual_matches_points[indices]
            #print(actual_matches_points.shape)


            if (actual_matches_points.shape[0] < 50):
                #print('FOUND NEW KEYFRAME!')
                actual_key_frame = list_frames[i]
                actual_matches_points = mkpts1_f_int
                list_key_frames.append(actual_key_frame)
                list_descriptors_key_frame.append(descriptors_img0)

                # Calculate chain rule for the actual key frame and store the obtained matrix in the correspondent list
                matrix_actual_key_frame = initial_matrix
                for index in range(i):
                    matrix_actual_key_frame = matrix_actual_key_frame @ list_homographies[i]
                matrices_key_frames.append(matrix_actual_key_frame)


        # print('Duration keyframe extraction: ',datetime.now() - start)

        # Conversion of points to opencv keypoints
        mkpts0_f_np = mkpts0_f.cpu().detach().numpy()
        mkpts1_f_np = mkpts1_f.cpu().detach().numpy()
        mconf_f_np = mconf_f.cpu().detach().numpy()
        h_matrix, inliers = cv2.estimateAffinePartial2D(mkpts1_f_np, mkpts0_f_np)
        #print(h_matrix.shape)
        row_to_concat = np.array([[0, 0, 1]])
        #print(row_to_concat.shape)
        h_matrix = np.concatenate((h_matrix, row_to_concat), axis=0)
        list_homographies.append(h_matrix)

        keypoints_img0 = convert_LoFTR_points_to_keypoints(mkpts0_f_np, mconf_f_np)
        keypoints_img1 = convert_LoFTR_points_to_keypoints(mkpts1_f_np, mconf_f_np)

    #print(list_key_frames)
    print('Number of key frames found: ', len(list_key_frames))
    return list_key_frames, mkpts0_f_int, mkpts1_f_int, keypoints_img0, keypoints_img1, list_homographies, matrices_key_frames

# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask):
    path_panorama_image_folder = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/output_panorama_images/SLAM_LoFTR')
    if not os.path.exists(path_panorama_image_folder):
        print('creating new panorama image folder')
        os.makedirs(path_panorama_image_folder)

    path_panorama_video_folder = os.path.join(os.getcwd(),
                                              '../dataset_MICCAI_2020_files/output_panorama_video/SLAM_LoFTR')
    if not os.path.exists(path_panorama_video_folder):
        print('creating new panorama video folder')
        os.makedirs(path_panorama_video_folder)

    path_panorama_image = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/output_panorama_images/SLAM_LoFTR/panorama_' + name_video + '.png')
    if not os.path.exists(path_panorama_image):
        black_canvas = np.zeros((2000, 2000, 3), dtype="uint8")
        panorama = np.copy(black_canvas)

        mask = np.uint8(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        #parameters for the video
        size = (black_canvas.shape[1], black_canvas.shape[0])
        fps = 25
        path_video = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_video/SLAM_LoFTR', name_video + '.mp4')
        out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

        for name in list_images:
            if('anon' not in name):
                list_images.remove(name)

        print('panorama creation')
        for index, image_name in enumerate(list_images):
            #print(os.path.join(os.getcwd(),path_images,image_name+'.png'))
            image = cv2.imread(os.path.join(os.getcwd(),path_images,image_name),1)
            image = cv2.resize(image, (NEW_SHAPE, NEW_SHAPE))
            #image_circle = cv2.bitwise_and(image, mask*255)
            image_circle = cv2.bitwise_and(image, mask)
            erosion_size = 10
            erosion_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
            mask_eroded = cv2.erode(mask, element)


            canvas_center = np.array([[np.cos(0), np.sin(0), black_canvas.shape[1]/2], [-np.sin(0), np.cos(0), black_canvas.shape[0]/2], [0, 0, 1]])


            if (index == 0):
                img_origin = np.array([[np.cos(0), np.sin(0), -image_circle.shape[1]/2], [-np.sin(0), np.cos(0), -image_circle.shape[0]/2], [0, 0, 1]])
                H4p = img_origin @ canvas_center


            else:
                H4p_hat = list_matrices[index-1]
                H4p = H4p_prev @ H4p_hat
                #print(H4p)

            panorama_curr = cv2.warpPerspective(image_circle, np.float32(H4p), (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)

            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(H4p), (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
            np.copyto(panorama, panorama_curr, where=mask_copy.astype(bool))

            edges = cv2.Canny(mask_copy, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) #edges with 3 channels

            dilatation_size = 3
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                               (dilatation_size, dilatation_size))
            dilatation = cv2.dilate(edges, element)


            panorama_with_border = np.copy(panorama)

            np.copyto(panorama_with_border, dilatation, where=dilatation.astype(bool))
            H4p_prev = H4p
            out.write(panorama_with_border)

        out.release()
        path_panorama = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/output_panorama_images/SLAM_LoFTR/','panorama_' + name_video + '.png')
        print(path_panorama)
        cv2.imwrite(path_panorama, panorama_with_border)
        cv2.destroyAllWindows()

        return panorama_with_border

    else:
        print('Panorama already reconstructed')
        panorama = cv2.imread(path_panorama_image, 1)
        return panorama


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
    image = image[crop_red:image.shape[1]-crop_red, crop_red:image.shape[0]-crop_red]
    image = cv2.resize(image, (shape, shape))

    # lines to add a black patch in the centre
    patch = np.zeros((patch_dim,patch_dim,3), dtype='uint8')
    x_offset = (image.shape[0]-patch.shape[0])//2
    y_offset = (image.shape[1]-patch.shape[1])//2
    image[y_offset:y_offset+patch.shape[1], x_offset:x_offset+patch.shape[0]] = patch
    return image

def relocalize (frame_to_relocalize_transf, list_key_frames, matcher):
    list_dist = []
    frame_to_relocalize_transf = cv2.cvtColor(frame_to_relocalize_transf, cv2.COLOR_RGB2GRAY)
    count = 0
    for key_frame in list_key_frames:
        print(count)
        count = count +1
        key_frame = cv2.cvtColor(key_frame, cv2.COLOR_RGB2GRAY)
        image_pair = [frame_to_relocalize_transf, key_frame]
        image, mkpts0_f, mkpts1_f, mconf_f, descriptors_img0, descriptors_img1, feature_space0, feature_space1 = LoFTR_pair(image_pair, mask, matcher)
        dist = np.linalg.norm(feature_space0 - feature_space1)
        list_dist.append(dist)

    min_dist = min(list_dist)
    min_index = list_dist.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]
    tuple_result = (min_index, key_frame_min_dist)
    return tuple_result


# It performs the sanity check  between the frame to relocalized with an applied transformation and the found keyframes
def sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama):
    list_key_frames = []
    panorama = panorama * 0.5

    for i in range(len(list_key_frames_names)):
        path_key_frame = path_frames+'/'+list_key_frames_names[i]
        key_frame = cv2.imread(path_key_frame, 1)
        key_frame = cv2.resize(key_frame, (NEW_SHAPE, NEW_SHAPE))
        list_key_frames.append(key_frame)

    tuple_result = relocalize(frame_to_relocalize_transf, list_key_frames, matcher)

    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    key_frame_warped = cv2.warpPerspective(tuple_result[1], np.float32(matrices_key_frames[tuple_result[0]]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_key_frame = cv2.warpPerspective(mask, np.float32(matrices_key_frames[tuple_result[0]]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_key_frame_copy = np.copy(mask_key_frame)

    panorama_with_key_frame = np.copy(panorama)
    np.copyto(panorama_with_key_frame, key_frame_warped, where=mask_key_frame.astype(bool))

    edges = cv2.Canny(mask_key_frame_copy*255, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
    dilatation_size = 5
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    dilatation = cv2.dilate(edges, element)
    b, g, r = cv2.split(dilatation)
    b[b == 255] = 0
    g[g == 255] = 0
    dilatation = cv2.merge([b, g, r])   # dilatation with red color
    np.copyto(panorama_with_key_frame, dilatation, where=dilatation.astype(bool))
    cv2.imwrite(os.path.join(os.getcwd(), 'panorama_key_frame.png'), panorama_with_key_frame)


    image_to_check_warped = cv2.warpPerspective(frame_to_relocalize_transf, np.float32(matrix_frame_to_relocalize), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_image_to_check = cv2.warpPerspective(mask, np.float32(matrix_frame_to_relocalize), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_image_to_check_copy = np.copy(mask_image_to_check)

    np.copyto(panorama_with_key_frame, image_to_check_warped, where=mask_image_to_check.astype(bool))
    edges = cv2.Canny(mask_image_to_check_copy*255, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
    dilatation_size = 5
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    dilatation = cv2.dilate(edges, element)
    b, g, r = cv2.split(dilatation)
    r[r == 255] = 0
    g[g == 255] = 0
    dilatation = cv2.merge([b, g, r])   # dilatation with red color
    np.copyto(panorama_with_key_frame, dilatation, where=dilatation.astype(bool))

    path_folder = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/sanity_check_panorama', name_experiment)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    path_folder = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/sanity_check_panorama', name_experiment, name_video)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    cv2.imwrite(os.path.join(path_folder, string_name_panorama), panorama_with_key_frame)
    return tuple_result









# MAIN FLOW
start = datetime.now()
NEW_SHAPE = 448
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
name_experiment = 'SLAM_LoFTR' #string containing the name used to identify the experiment

list_videos = sorted(os.listdir(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset')))   # contains list of videos names

matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load(os.path.join(os.getcwd(), "../code_Git_imported/LoFTR/weights/outdoor_ds.ckpt"))['state_dict'])
matcher = matcher.eval().to(device)

index_column = 0
index_row = 0

experiment_result = np.zeros((18,len(list_videos)), dtype=object)

for name_video in list_videos:
    print(name_video)
    path_frames = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images')
    #mask = mask/255. #binary mask
    mask = cv2.imread(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'mask.png'), 0)
    mask = cv2.resize(mask, (NEW_SHAPE, NEW_SHAPE))
    shape_image = mask.shape[0]
    list_frames = sorted(os.listdir(path_frames)) # list containing all the images names

    black_canvas = np.zeros((2000, 2000, 3), dtype="uint8")
    canvas_center = np.array([[np.cos(0), np.sin(0), black_canvas.shape[1] / 2], [-np.sin(0), np.cos(0), black_canvas.shape[0] / 2],[0, 0, 1]])
    img_origin = np.array([[np.cos(0), np.sin(0), -mask.shape[1] / 2], [-np.sin(0), np.cos(0), -mask.shape[0] / 2], [0, 0, 1]])
    initial_matrix = img_origin @ canvas_center

    list_key_frames_names, mkpts0_f_int, mkpts1_f_int, keypoints_img0, keypoints_img1, list_homografies, matrices_key_frames = compute_key_frames(list_frames, mask, matcher, initial_matrix, name_video)
    panorama = panorama_reconstruction(list_frames, list_homografies, path_frames, name_video, mask)


    num_frame_to_relocalize = 50
    name_frame_to_relocalize = list_frames[num_frame_to_relocalize]
    path_frame_to_relocalize = os.path.join(path_frames, name_frame_to_relocalize)
    matrix_frame_to_relocalize = initial_matrix
    for i in range(num_frame_to_relocalize):
        matrix_frame_to_relocalize = matrix_frame_to_relocalize @ list_homografies[i]


    #print(path_frame_to_relocalize)
    frame_to_relocalize = cv2.imread(path_frame_to_relocalize, 1)
    frame_to_relocalize = cv2.resize(frame_to_relocalize, (NEW_SHAPE, NEW_SHAPE))

    # LIST OF ALL THE SANITY CHECKS

    # sanity check with no transformation applied
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    #print(frame_to_relocalize_transf.shape)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[0, index_column] = tuple_result

    # sanity check with 10 degrees rotation
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 10
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[1, index_column] = tuple_result

    # sanity check with 30 degrees rotation
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 30
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[2, index_column] = tuple_result

    # sanity check with 60 degrees rotation
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 60
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[3, index_column] = tuple_result

    # sanity check with contrast alpha=0.8
    ill_alpha = 0.80
    ill_beta = 1.0
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[4, index_column] = tuple_result

    # sanity check with contrast alpha=0.9
    ill_alpha = 0.90
    ill_beta = 1.0
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[5, index_column] = tuple_result

    # sanity check with contrast alpha=1.10
    ill_alpha = 1.10
    ill_beta = 1.0
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[6, index_column] = tuple_result

    # sanity check with contrast alpha=1.20
    ill_alpha = 1.20
    ill_beta = 1.0
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[7, index_column] = tuple_result

    # sanity check with contrast beta=0.8
    ill_alpha = 1.0
    ill_beta = 0.8
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[8, index_column] = tuple_result

    # sanity check with contrast beta=0.9
    ill_alpha = 1.0
    ill_beta = 0.9
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[9, index_column] = tuple_result

    # sanity check with contrast beta=1.10
    ill_alpha = 1.0
    ill_beta = 1.10
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[10, index_column] = tuple_result

    # sanity check with contrast beta=1.20
    ill_alpha = 1.0
    ill_beta = 1.20
    rot = 0
    crop_red = 0
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[11, index_column] = tuple_result

    # sanity check crop reduction=20
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 0
    crop_red = 20
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[12, index_column] = tuple_result

    # sanity check crop reduction=30
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 0
    crop_red = 30
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[13, index_column] = tuple_result

    # sanity check crop reduction=50
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 0
    crop_red = 50
    patch_dim = 0
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[14, index_column] = tuple_result

    # sanity check with patch insertion of patch_dim=100
    ill_alpha = 1.0
    ill_beta = 1.0
    rot = 0
    crop_red = 0
    patch_dim = 100
    frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
    tuple_result = sanity_check(frame_to_relocalize_transf, list_key_frames_names, path_frames, matcher, panorama, mask, matrices_key_frames, matrix_frame_to_relocalize, string_name_panorama)
    experiment_result[15, index_column] = tuple_result

    print(type(list_key_frames_names[0]))
    # code to add the interval of key frames between which we can find the frame to reattach
    list_key_frames_num = [item[8:] for item in list_key_frames_names]
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

    for i in reversed(range(len(list_key_frames_names))):
        if frame_to_reattach_num < list_tuples_frames_num[i][0]:
            max_interval = i

    interval = (min_interval, max_interval)
    tuple_frame_to_reattach = (frame_to_relocalize, interval)

    experiment_result[16, index_column] = tuple_frame_to_reattach
    experiment_result[17, index_column] = list_key_frames_names

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
path_folder_experiments = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files', 'experiments_files')
if not os.path.exists(path_folder_experiments):
    os.makedirs(path_folder_experiments)

filepath = os.path.join(path_folder_experiments, name_experiment + '.xlsx')
df.to_excel(filepath, index=True)
print('Duration keyframe extraction: ',datetime.now() - start)


















































