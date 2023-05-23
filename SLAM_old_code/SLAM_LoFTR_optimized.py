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
#np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime
from torch.multiprocessing import Manager
import torch.multiprocessing as mp
from code_Git_imported.LoFTR.src.loftr import LoFTR, default_cfg

# Global variables
NEW_SHAPE = 448
CONFIDENCE_THRESHOLD = 0.75
NUM_POINTS_THRESHOLD = 20
CANVAS_SHAPE = 2000
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
name_experiment = 'SLAM_LoFTR_optimized'  # string containing the name used to identify the experiment

canvas_center = np.array([[np.cos(0), np.sin(0), CANVAS_SHAPE / 2], [-np.sin(0), np.cos(0), CANVAS_SHAPE / 2], [0, 0, 1]])
img_origin = np.array([[np.cos(0), np.sin(0), -NEW_SHAPE / 2], [-np.sin(0), np.cos(0), -NEW_SHAPE / 2],[0, 0, 1]])
INITIAL_MATRIX = img_origin @ canvas_center



def build_video_key_frame(list_key_frames, name_video, path_frames, mask, list_homographies, list_frames):
    black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
    panorama = np.copy(black_canvas)
    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    key_frames_names= [x[0] for x in list_key_frames]

    # parameters for the video
    size = (black_canvas.shape[1], black_canvas.shape[0])
    fps = 25
    path_video = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_video', name_experiment,
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
            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(list_key_frames[index_kf][1]),(panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
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
                for i in range(index-1):
                    H = H @ list_homographies[i]



            erosion_size = 10
            erosion_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                                (erosion_size, erosion_size))
            mask_eroded = cv2.erode(mask, element)
            # panorama_curr = cv2.warpPerspective(image_circle, np.float32(key_frame[1]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(H),(panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
            regions = measure.regionprops(mask_copy)
            for props in regions:
                centroid = (np.array(props.centroid)).astype(int)
            edges = cv2.Canny(mask_copy, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels

            dilatation_size = 1
            dilation_shape = cv2.MORPH_ELLIPSE
            element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),(dilatation_size, dilatation_size))
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
    path_panorama = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_images', name_experiment,
                                 'panorama_' + name_video + '_key_frames.png')
    # print(path_panorama)
    cv2.imwrite(path_panorama, panorama)
    cv2.destroyAllWindows()
    print('key frames video done!')








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
    #img1_raw = img1_raw.cpu().detach().numpy()

    # Delete points with a low confidence
    mconf_new = mconf<CONFIDENCE_THRESHOLD
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


    tuple_for_queue = (mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1, index_iteration, list_frames, seq_is_end)
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


def convert_LoFTR_points_to_keypoints(pts, confidence):
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        for i in range(len(pts)):
            keypoint = cv2.KeyPoint(pts[i][0], pts[i][1], size=1, response=confidence[i])
            kps.append(keypoint)
    return kps



def refine_key_frames(list_key_frames):
    print('Key frames before refinement: ', len(list_key_frames))
    list_dist = []
    print(len(list_key_frames))
    list_indexes_remove = []
    for i in range(len(list_key_frames)-1):
        for j in range(i+1,len(list_key_frames)-1,1):
            dist = np.linalg.norm(list_key_frames[i][2]-list_key_frames[j][2])
            #dist = torch.cdist(LIST_KEY_FRAMES[i][2],LIST_KEY_FRAMES[j][2],p=2)
            # print('Coppie uguali')
            # print(list_key_frames[i][0])
            # print(list_key_frames[j][0])
            # print(dist)
            list_dist.append(dist)
            if dist<1400:
                list_indexes_remove.append(list_key_frames[i][0])

    for element in list_key_frames:
        if (element[0] in list_indexes_remove):
            list_key_frames.remove(element)
    print('Key frames after refinement: ', len(list_key_frames))
    return list_key_frames


# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask, name_experiment):
    path_panorama_image_folder = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_images', name_experiment)
    if not os.path.exists(path_panorama_image_folder):
        #print('creating new panorama image folder')
        os.makedirs(path_panorama_image_folder)

    path_panorama_video_folder = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_video', name_experiment)
    if not os.path.exists(path_panorama_video_folder):
        #print('creating new panorama video folder')
        os.makedirs(path_panorama_video_folder)

    path_panorama_image = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_images', name_experiment, 'panorama_' + name_video + '.png')
    if not os.path.exists(path_panorama_image):
        black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
        panorama = np.copy(black_canvas)
        panorama_mask = np.copy(black_canvas)

        mask = np.uint8(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        #parameters for the video
        size = (black_canvas.shape[1], black_canvas.shape[0])
        fps = 25
        path_video = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_video', name_experiment, name_video + '.mp4')
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
                #print(index-1)
                H4p_hat = list_matrices[index-1]
                H4p = H4p_prev @ H4p_hat
                #print(H4p)




            panorama_curr = cv2.warpPerspective(image_circle, np.float32(H4p), (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)

            mask_copy = cv2.warpPerspective(mask_eroded, np.float32(H4p), (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)

            merge_mertens = cv2.createMergeMertens()
            intersection_mask = cv2.bitwise_and(panorama_mask,mask_copy)
            mask_inv = 255 - intersection_mask;
            intersection_panorama = mask_inv * panorama
            intersection_panorama_curr= mask_inv*panorama_curr

            np.copyto(panorama, panorama_curr, where=mask_copy.astype(bool))
            np.copyto(panorama_mask, mask_copy, where=mask_copy.astype(bool))

            # edges = cv2.Canny(mask_copy, 100, 200)
            # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) #edges with 3 channels
            # dilatation_size = 3
            # dilation_shape = cv2.MORPH_ELLIPSE
            # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
            #                                    (dilatation_size, dilatation_size))
            # dilatation = cv2.dilate(edges, element)
            # panorama_with_border = np.copy(panorama)
            # np.copyto(panorama_with_border, dilatation, where=dilatation.astype(bool))

            #print(intersection_panorama.shape)
            #print(intersection_panorama_curr.shape)
            #print(intersection_panorama.dtype)
            #print(intersection_panorama_curr.dtype)
            #img_list = [np.uint8(intersection_panorama), np.uint8(intersection_panorama_curr)]

            #res_mertens = merge_mertens.process(img_list)
            #panorama = res_mertens

            H4p_prev = H4p
            #out.write(np.uint8(res_mertens*255))
            out.write(panorama)

        out.release()
        path_panorama = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_images', name_experiment, 'panorama_' + name_video + '.png')
        #print(path_panorama)
        cv2.imwrite(path_panorama, panorama)
        cv2.destroyAllWindows()

        return panorama

    else:
        print('Panorama already reconstructed')
        panorama = cv2.imread(path_panorama_image, 1)
        return panorama


def check_key_frame(queue, lock, list_homographies, list_key_frames):
    # Synchronize access to the console
    SEQ_IS_END = False
    
    with lock:
        print('Starting consumer => {}'.format(os.getpid()))

    # # Run indefinitely
    loop = 0
    while SEQ_IS_END==False:

        loop = loop + 1
        lock.acquire()
        try:
            #print('sono nel try')
            dictionary_from_queue = queue.get(timeout=1) # che un po di english ci piace
            #print(type(dictionary_from_queue))

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
            h_matrix, inliers = cv2.estimateAffinePartial2D(mkpts1_f_np, mkpts0_f_np, descriptors_img0)
            row_to_concat = np.array([[0, 0, 1]])
            h_matrix = np.concatenate((h_matrix, row_to_concat), axis=0)
            list_homographies.append(h_matrix)

            #keypoints_img0 = convert_LoFTR_points_to_keypoints(mkpts0_f, mconf_f)
            #keypoints_img1 = convert_LoFTR_points_to_keypoints(mkpts1_f, mconf_f)

            if (i == 0):
                actual_key_frame = list_frames[i]
                H = np.eye(3)
                H_key_frame =  INITIAL_MATRIX @ H
                tuple_key_frame = (actual_key_frame, H_key_frame, feature_space0.cpu().detach().numpy(), descriptors_img0.cpu().detach().numpy())
                list_key_frames.append(tuple_key_frame)
                actual_matches_points= mkpts1_f_int

            else:

                # Find actual matches: keypoints from previous iterations that are present also in this iteration
                #pydevd_pycharm.settrace('10.79.251.35', port=8200, stdoutToServer=True, stderrToServer=True)
                #values, indices = torch.topk(((actual_matches_points.t() == mkpts0_f_int.unsqueeze(-1)).all(dim=1)).int(), 1, 1)
                #indices = indices[values != 0]
                indices = ((actual_matches_points[:, None] == mkpts0_f_int[None, :]).all(dim=2)).nonzero()
                #print(indices[:,0])
                actual_matches_points = actual_matches_points[indices[:,0]]
                print('Number of matches points after topk alternative: ', len(actual_matches_points))

                # actual_matches_points_copy = torch.clone(actual_matches_points)
                # for y in mkpts0_f_int:
                #     actual_matches_points_copy = actual_matches_points_copy[actual_matches_points_copy.eq(y).all(dim=1).logical_and()]
                # print('Number of matches points after logical and: ', len(actual_matches_points_copy))
                # pydevd_pycharm.settrace('localhost', port=8200, stdoutToServer=True, stderrToServer=True)



                if (actual_matches_points.shape[0] < NUM_POINTS_THRESHOLD):

                    print('New key frame found')
                    actual_key_frame = list_frames[i]

                    H_key_frame = INITIAL_MATRIX
                    for index in range(i):
                        H_key_frame = H_key_frame @ list_homographies[index-1]


                    tuple_key_frame = (actual_key_frame, H_key_frame, feature_space0.cpu().detach().numpy(), descriptors_img0.cpu().detach().numpy())
                    tuple_key_frame_copy = deepcopy(tuple_key_frame)
                    list_key_frames.append(tuple_key_frame_copy)
                    actual_matches_points = mkpts1_f_int
        finally:
            lock.release()


def compute_key_frames_mosaic (list_frames, mask, matcher, name_video):
    list_homographies = []
    list_homographies.append(np.eye(3))
    list_key_frames = []
    seq_isend = False

    mask = np.uint8(mask)
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Create the Queue object and the consumer
    queue = mp.Queue()
    lock = mp.Lock()
    #c = mp.spawn(check_key_frame, args=(queue, lock), nprocs=1, join=True, daemon=True, start_method='spawn')
    with Manager() as manager:

        list_homographies = manager.list(list_homographies)
        list_key_frames = manager.list(list_key_frames)

        c = mp.Process(target=check_key_frame, args=(queue, lock, list_homographies, list_key_frames))
        c.daemon = True
        c.start()

        t = trange(len(list_frames) - 1)
        # Key frame extraction
        for i in t:
            #print(i)
            start = datetime.now()
            if (i==len(t)-1):
                seq_isend = True
            img0_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images', list_frames[i])
            img1_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images', list_frames[i + 1])
            img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
            image_pair = [img0_raw, img1_raw]
            tuple_loftr = LoFTR_pair(image_pair, mask, matcher, i, list_frames, seq_isend)
            dict_from_tuple = [
                tuple_loftr[7],tuple_loftr[8], tuple_loftr[9],tuple_loftr[0],
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

def relocalize (frame_to_relocalize_transf_name, matcher, mask, list_key_frames):
    list_dist = []
    list_h_transf = []
    frame_to_relocalize_transf = cv2.cvtColor(frame_to_relocalize_transf_name, cv2.COLOR_RGB2GRAY)
    count = 0
    for key_frame in list_key_frames:
        count = count +1
        key_frame = cv2.cvtColor(key_frame, cv2.COLOR_RGB2GRAY)
        image_pair = [frame_to_relocalize_transf, key_frame]
        mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f, feature_space0, feature_space1 = LoFTR_pair_relocalization(image_pair, mask, matcher)

        dist = np.linalg.norm(feature_space0 - feature_space1)
        list_dist.append(dist)

        # mkpts0_f_np = mkpts0_f.cpu().detach().numpy()
        # mkpts1_f_np = mkpts1_f.cpu().detach().numpy()
        # if (frame_to_relocalize_transf_name[8:12] < key_frame[0][8:12]):
        #     h_matrix, inliers = cv2.estimateAffinePartial2D(mkpts1_f_np, mkpts0_f_np)
        # else:
        #     h_matrix, inliers = cv2.estimateAffinePartial2D(mkpts0_f_np, mkpts1_f_np)
        # row_to_concat = np.array([[0, 0, 1]])
        # h_matrix = np.concatenate((h_matrix, row_to_concat), axis=0)
        # list_h_transf.append(h_matrix)


    min_dist = min(list_dist)
    min_index = list_dist.index(min_dist)
    #h_min_index = list_h_transf[min_index]
    h_min_index = 0
    key_frame_min_dist = list_key_frames[min_index]
    tuple_result = (min_index, key_frame_min_dist, h_min_index)
    return tuple_result


def compute_metric(img1, img2, mat1, mat2, h_estim_1_2, path_frames, mask):
    black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")

    img1 = cv2.imread(os.path.join(path_frames, img1), 1)
    img2 = cv2.imread(os.path.join(path_frames, img2), 1)

    panorama_chain_rule = np.copy(black_canvas)
    img1_warped = cv2.warpPerspective(img1, np.float32(mat1), (panorama_chain_rule.shape[1], panorama_chain_rule.shape[0]),flags=cv2.INTER_NEAREST)
    mask_img1 = cv2.warpPerspective(mask, np.float32(mat1), (panorama_chain_rule.shape[1], panorama_chain_rule.shape[0]),flags=cv2.INTER_NEAREST)
    np.copyto(panorama_chain_rule, img1_warped, where=mask_img1.astype(bool))
    img2_warped = cv2.warpPerspective(img2, np.float32(mat2), (panorama_chain_rule.shape[1], panorama_chain_rule.shape[0]),flags=cv2.INTER_NEAREST)
    mask_img2 = cv2.warpPerspective(mask, np.float32(mat2), (panorama_chain_rule.shape[1], panorama_chain_rule.shape[0]),flags=cv2.INTER_NEAREST)
    np.copyto(panorama_chain_rule, img2_warped, where=mask_img2.astype(bool))

    panorama_kf_estim = np.copy(black_canvas)
    #img1_warped = cv2.warpPerspective(img1, np.float32(mat1), (panorama_kf_estim.shape[1], panorama_kf_estim.shape[0]),flags=cv2.INTER_NEAREST)
    #mask_img1 = cv2.warpPerspective(mask, np.float32(mat1), (panorama_kf_estim.shape[1], panorama_kf_estim.shape[0]),flags=cv2.INTER_NEAREST)
    np.copyto(panorama_kf_estim, img1_warped, where=mask_img1.astype(bool))
    mat2_estim = mat1 @ h_estim_1_2
    img2_warped = cv2.warpPerspective(img2, np.float32(mat2_estim), (panorama_kf_estim.shape[1], panorama_kf_estim.shape[0]),flags=cv2.INTER_NEAREST)
    mask_img2 = cv2.warpPerspective(mask, np.float32(mat2_estim), (panorama_kf_estim.shape[1], panorama_kf_estim.shape[0]),flags=cv2.INTER_NEAREST)
    np.copyto(panorama_kf_estim, img2_warped, where=mask_img2.astype(bool))








# It performs the sanity check  between the frame to relocalized with an applied transformation and the found keyframes
def sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames):
    list_key_frames_images = []
    panorama = panorama * 0.5

    for i in range(len(list_key_frames)):
        path_key_frame = path_frames+'/'+ list_key_frames[i][0]
        key_frame = cv2.imread(path_key_frame, 1)
        key_frame = cv2.resize(key_frame, (NEW_SHAPE, NEW_SHAPE))
        list_key_frames_images.append(key_frame)

    tuple_result = relocalize(frame_to_relocalize_transf, matcher, mask, list_key_frames_images)

    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # METRIC CALCULATION
    # 1) obtain order between frame to relocalize and nearest key frame
    # if(frame_to_relocalize_transf[8:12] > tuple_result[1][8:12]):
    #     img1 = tuple_result[1]
    #     mat1 = list_key_frames[tuple_result[0]][1]
    #     img2 = frame_to_relocalize_transf
    #     mat2 = matrix_frame_to_relocalize
    # else:
    #     img1 = frame_to_relocalize_transf
    #     mat1 = matrix_frame_to_relocalize
    #     img2 = tuple_result[1]
    #     mat2 = list_key_frames[tuple_result[0]][1]
    # h_estim_1_2 = tuple_result[2]
    # metric = compute_metric(img1, img2, mat1, mat2, h_estim_1_2, path_frames, mask)






    # RESULT REPRESENTATION
    ind = tuple_result[0]

    key_frame_warped = cv2.warpPerspective(tuple_result[1], np.float32(list_key_frames[tuple_result[0]][1]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_key_frame = cv2.warpPerspective(mask, np.float32(list_key_frames[tuple_result[0]][1]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_key_frame_copy = np.copy(mask_key_frame)
    panorama_with_key_frame = np.copy(panorama)
    np.copyto(panorama_with_key_frame, key_frame_warped, where=mask_key_frame.astype(bool))

    edges = cv2.Canny(mask_key_frame_copy, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
    dilatation_size = 10
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    dilatation = cv2.dilate(edges, element)
    b, g, r = cv2.split(dilatation)
    b[b == 255] = 0
    g[g == 255] = 0
    dilatation = cv2.merge([b, g, r])   # dilatation with red color
    np.copyto(panorama_with_key_frame, dilatation, where=dilatation.astype(bool))


    image_to_check_warped = cv2.warpPerspective(frame_to_relocalize_transf, np.float32(matrix_frame_to_relocalize), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_image_to_check = cv2.warpPerspective(mask, np.float32(matrix_frame_to_relocalize), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_image_to_check_copy = np.copy(mask_image_to_check)

    np.copyto(panorama_with_key_frame, image_to_check_warped, where=mask_image_to_check.astype(bool))
    edges = cv2.Canny(mask_image_to_check_copy, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # edges with 3 channels
    dilatation_size = 10
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
    tuple_to_return = (tuple_result[0], list_key_frames[tuple_result[0]][0])
    return tuple_to_return











# MAIN FLOW
def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


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

        path_matrices = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'output_sp_RANSAC')
        list_images = sorted(os.listdir(path_frames))  # list containing all the images names
        list_matrices = sorted(os.listdir(path_matrices))  # list containing all the matrices files names

        list_images_no_extension = []  # list containing all the images names without the extension .png
        list_matrices_no_extension = []  # list containing all the matrices files names without the extension .txt

        for name_image in list_images:
            name_image = name_image.replace('.png', '')
            list_images_no_extension.append(name_image)

        for name_matrix in list_matrices:
            name_matrix = name_matrix.replace('.txt', '')
            list_matrices_no_extension.append(name_matrix)

        list_images_no_extension.sort()
        list_matrices_no_extension.sort()

        list_images_common = []  # list containing all the names of images with corresponding matrices
        for name in list_images_no_extension:
            if name in list_matrices_no_extension:
                list_images_common.append(name)

        list_matrices_names_common = []  # list containing all the names of matrices with corresponding images
        for name in list_matrices_no_extension:
            if name in list_images_no_extension:
                list_matrices_names_common.append(name)

        list_images_common.sort()
        list_matrices_names_common.sort()

        # list_images_common contains all the images names with correspondent matrices
        # path_images is the path to the images folder
        # path_matrices is the path to the matrices folder

        list_frames = []
        for i in list_images_common:
            for j in list_images:
                if i in j:
                    list_frames.append(j)



        list_homographies, list_key_frames = compute_key_frames_mosaic(list_frames, mask, matcher, name_video)
        list_key_frames = refine_key_frames(list_key_frames)

        build_video_key_frame(list_key_frames, name_video, path_frames, mask, list_homographies, list_frames)

        panorama = panorama_reconstruction(list_frames, list_homographies, path_frames, name_video, mask, name_experiment)

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

        # sanity check with no transformation applied
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[0, index_column] = tuple_result

        # sanity check with 10 degrees rotation
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 10
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[1, index_column] = tuple_result

        # sanity check with 30 degrees rotation
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 30
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[2, index_column] = tuple_result

        # sanity check with 60 degrees rotation
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 60
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[3, index_column] = tuple_result

        # sanity check with contrast alpha=0.8
        ill_alpha = 0.80
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[4, index_column] = tuple_result

        # sanity check with contrast alpha=0.9
        ill_alpha = 0.90
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[5, index_column] = tuple_result

        # sanity check with contrast alpha=1.10
        ill_alpha = 1.10
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[6, index_column] = tuple_result

        # sanity check with contrast alpha=1.20
        ill_alpha = 1.20
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[7, index_column] = tuple_result

        # sanity check with contrast beta=0.8
        ill_alpha = 1.0
        ill_beta = 0.8
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[8, index_column] = tuple_result

        # sanity check with contrast beta=0.9
        ill_alpha = 1.0
        ill_beta = 0.9
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[9, index_column] = tuple_result

        # sanity check with contrast beta=1.10
        ill_alpha = 1.0
        ill_beta = 1.10
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[10, index_column] = tuple_result

        # sanity check with contrast beta=1.20
        ill_alpha = 1.0
        ill_beta = 1.20
        rot = 0
        crop_red = 0
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[11, index_column] = tuple_result

        # sanity check crop reduction=20
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 20
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[12, index_column] = tuple_result

        # sanity check crop reduction=30
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 30
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[13, index_column] = tuple_result

        # sanity check crop reduction=50
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 50
        patch_dim = 0
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[14, index_column] = tuple_result

        # sanity check with patch insertion of patch_dim=100
        ill_alpha = 1.0
        ill_beta = 1.0
        rot = 0
        crop_red = 0
        patch_dim = 100
        frame_to_relocalize_transf = add_transformation(frame_to_relocalize, ill_alpha, ill_beta, rot, crop_red, patch_dim)
        string_name_panorama = 'sc_rot' + str(rot) + '_contr' + str(ill_alpha) + '_bright' + str(ill_beta) + '_crop' + str(crop_red) + '_patch' + str(patch_dim) + '.png'
        tuple_result = sanity_check(frame_to_relocalize_transf, path_frames, matcher, panorama, mask, matrix_frame_to_relocalize, string_name_panorama, name_video, list_key_frames)
        experiment_result[15, index_column] = tuple_result

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
    path_folder_experiments = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files', 'experiments_files')
    if not os.path.exists(path_folder_experiments):
        os.makedirs(path_folder_experiments)

    filepath = os.path.join(path_folder_experiments, name_experiment + '.xlsx')
    df.to_excel(filepath, index=True)



if __name__=='__main__':
    main()
























