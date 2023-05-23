import itertools
import os
import sys
import kornia
import torch
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.cm as cm

from code_Git_imported.LoFTR.src.utils.plotting import make_matching_figure
from code_Git_imported.LoFTR.src.loftr import LoFTR, default_cfg

NEW_SHAPE = 448


def LoFTR_pair(image_pair, mask, matcher):
    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (NEW_SHAPE, NEW_SHAPE))
    img1_raw = cv2.resize(img1_raw, (NEW_SHAPE, NEW_SHAPE))
    mask = cv2.resize(mask, (NEW_SHAPE, NEW_SHAPE))
    mask[mask != 0] = 255.
    img0_raw = cv2.bitwise_and(img0_raw, mask)
    img1_raw = cv2.bitwise_and(img1_raw, mask)

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    img0 = kornia.filters.gaussian_blur2d(img0, (9, 9), (1.5, 1.5))
    img1 = kornia.filters.gaussian_blur2d(img1, (9, 9), (1.5, 1.5))
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction all in pytorch
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f']
        mkpts1 = batch['mkpts1_f']
        mconf = batch['mconf'].cpu()
        descriptors_img0_tensor = matcher.des0
        descriptors_img1_tensor = matcher.des1

    descriptors_img0 = descriptors_img0_tensor
    descriptors_img1 = descriptors_img1_tensor


    # # Inference with LoFTR and get prediction --> in torch + numpy
    # with torch.no_grad():
    #     matcher(batch)
    #     mkpts0 = batch['mkpts0_f'].cpu().numpy()
    #     mkpts1 = batch['mkpts1_f'].cpu().numpy()
    #     mconf = batch['mconf'].cpu().numpy()
    #     descriptors_img0_tensor = matcher.des0
    #     descriptors_img1_tensor = matcher.des1
    #
    # descriptors_img0 = descriptors_img0_tensor.cpu().detach().numpy()
    # descriptors_img1 = descriptors_img1_tensor.cpu().detach().numpy()



    # CONTROLLARE QUI
    img0_raw = img0.squeeze()
    #img0_raw = img0_raw.cpu().detach().numpy()
    img1_raw = img1.squeeze()
    #img1_raw = img1_raw.cpu().detach().numpy()


    # Draw
    color = cm.jet(mconf, alpha=0.7)

    # Delete points with a low confidence
    for elem in mconf:
        if (elem) < 0.75:
            index = np.where(mconf == elem)
            mconf = np.delete(mconf, index, axis=0)
            mkpts0 = np.delete(mkpts0, index, axis=0)
            mkpts1 = np.delete(mkpts1, index, axis=0)
            color = np.delete(color, index, axis=0)
            descriptors_img0 = np.delete(descriptors_img0, index, axis=0)
            descriptors_img1 = np.delete(descriptors_img1, index, axis=0)


    # Calculation of eroded mask
    erosion_size = 10
    erosion_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))
    mask_eroded = cv2.erode(mask, element)

    # Delete points in the erosion area of mkpts0
    mconf_f = np.empty((0), np.float32)
    mkpts0_f = np.empty((0, 2), np.uint8)
    mkpts1_f = np.empty((0, 2), np.uint8)
    color_f = np.empty((0, 4), np.float32)
    descriptors_img0_f = np.empty((0, 256), np.float32)
    descriptors_img1_f = np.empty((0, 256), np.float32)
    for i, elem in enumerate(mkpts0):
        #print(elem)
        coord_0 = int(elem[0])
        coord_1 = int(elem[1])
        if mask_eroded[coord_0, coord_1] > 0:
            mconf_f = np.append(mconf_f, np.array([mconf[i]]), axis=0)
            mkpts0_f = np.append(mkpts0_f, np.array([mkpts0[i]]), axis=0)
            mkpts1_f = np.append(mkpts1_f, np.array([mkpts1[i]]), axis=0)
            color_f = np.append(color_f, np.array([color[i]]), axis=0)
            descriptors_img0_f= np.append(descriptors_img0_f, np.array([descriptors_img0[i]]), axis=0)
            descriptors_img1_f= np.append(descriptors_img1_f, np.array([descriptors_img1[i]]), axis=0)

    text = ['LoFTR', 'Matches: {}'.format(len(mkpts0))]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0_f, mkpts1_f, color_f, mkpts0_f, mkpts1_f, text)

    fig.tight_layout(pad=0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image, mkpts0_f, mkpts1_f, mconf_f, descriptors_img0_f, descriptors_img1_f



def convert_LoFTR_points_to_keypoints(pts, confidence):
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        for i in range(len(pts)):
            keypoint = cv2.KeyPoint(pts[i][0], pts[i][1], size=1, response=confidence[i])
            kps.append(keypoint)
    return kps




# MAIN FLOW
list_frames = sorted(os.listdir(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/images')))
mask = cv2.imread(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/mask.png'), 0)

list_key_frames = []
list_descriptors_key_frame = []
# fps = 25.0
# path_video = os.path.join(os.getcwd(), 'ouput_LOFTR_anon001.mp4')
# out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (750, 450), True)

# Key frame extraction and loop closure check (MANCA LOOP CLOSURE)
for i in range(len(list_frames)-1):
    print(i)
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(os.path.join(os.getcwd(), "../code_Git_imported/LoFTR/weights/outdoor_ds.ckpt"))['state_dict'])
    matcher = matcher.eval().cuda()

    img0_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/images', list_frames[i])
    img1_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/images', list_frames[i + 1])
    image_pair = [img0_pth, img1_pth]

    image, mkpts0_f, mkpts1_f,mconf_f, descriptors_img0, descriptors_img1 = LoFTR_pair(image_pair, mask, matcher)
    mkpts0_f_int = np.around(mkpts0_f)
    mkpts1_f_int = np.around(mkpts1_f)
    #out.write(image)
    #np.savetxt('First_'+str(i)+'.txt', mkpts0_f_int, delimiter=',', fmt='%1.4f')
    #np.savetxt('Second_'+str(i)+'.txt', mkpts1_f_int, delimiter=',', fmt='%1.4f')

    keypoints_img0 = convert_LoFTR_points_to_keypoints(mkpts0_f, mconf_f)
    keypoints_img1 = convert_LoFTR_points_to_keypoints(mkpts1_f, mconf_f)

    if (i==0):
        actual_key_frame = list_frames[i]
        list_key_frames.append(actual_key_frame)
        matches_key_frame = mkpts1_f_int
        list_descriptors_key_frame.append(descriptors_img0)

    else:
        # Find actual matches: keypoints from previous iterations that are present also in this iteration
        actual_matches_points = np.empty((0, 2))
        for i0, i1 in itertools.product(np.arange(mkpts0_f_int.shape[0]), np.arange(matches_key_frame.shape[0])):
            if np.all(np.isclose(mkpts0_f_int[i0], matches_key_frame[i1], atol=1.00)):
                actual_matches_points = np.concatenate((actual_matches_points, [mkpts0_f_int[i0]]), axis=0)

        if (i == 1):
            num_matches_points_last_key_frame = actual_matches_points.shape[0]

        #print('shape of actual_matches_points: ', actual_matches_points.shape)
        #print(actual_matches_points)
        if (actual_matches_points.shape[0] < 50) :
            #print('FOUND NEW KEYFRAME!')
            actual_key_frame = list_frames[i]
            list_key_frames.append(actual_key_frame)
            matches_key_frame = mkpts0_f_int
            list_descriptors_key_frame.append(descriptors_img0)


#out.release()
#cv2.destroyAllWindows()
print(list_key_frames)
print('Number of key frames found: ', len(list_key_frames))


# RILOCALIZATION --> da riscrivere

list_len_points_matching = []
frame_to_relocalize = list_frames[50]
for i in range(len(list_key_frames)-1):
    frame_to_relocalize_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/images', frame_to_relocalize)
    frame_pth = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/images', list_key_frames[i])
    image_pair = [frame_to_relocalize_pth, frame_pth]

    print(frame_to_relocalize)
    print(list_key_frames[i])

    image, mkpts0_f, mkpts1_f,mconf_f, descriptors_img0, descriptors_img1 = LoFTR_pair(image_pair, mask, matcher)
    list_len_points_matching.append(len(mkpts0_f))

max_value = max(list_len_points_matching)
max_index = list_len_points_matching.index(max_value)
print('frame to relocalize: ', frame_to_relocalize)
print('key frame with max number of match: ', list_frames[max_index])


