from datetime import datetime
import imutils
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import loadtxt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import interpolate
import torch
from torch import optim, nn
from torchvision import models, transforms
from matplotlib.animation import FuncAnimation
import pandas as pd
import segmentation_models_pytorch as smp
#import pydevd_pycharm
from skimage.metrics import structural_similarity as ssim
from skimage import measure


name_experiment = 'ResNet_FINAL_ALL_VIDEOS'   #string containing the name used to identify the experiment
DATASET_PATH = os.path.join(os.getcwd(), 'final_dataset')
DATASET_FILES_PATH = os.path.join(os.getcwd(), 'final_dataset_files')
CANVAS_SHAPE = 4000
IMAGE_SHAPE = 470
canvas_center = np.array(
    [[np.cos(0), np.sin(0), CANVAS_SHAPE / 2], [-np.sin(0), np.cos(0), CANVAS_SHAPE / 2], [0, 0, 1]])
img_origin = np.array([[np.cos(0), np.sin(0), -IMAGE_SHAPE / 2], [-np.sin(0), np.cos(0), -IMAGE_SHAPE / 2], [0, 0, 1]])
INITIAL_MATRIX = img_origin @ canvas_center



# Class of the used network, derived from a RESNET
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract RESNET-50 Feature Layers
        self.modules = list(model.children())[:-1]
        self.features = nn.Sequential(*self.modules)


    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        #out = self.modules(x)   # 64x512x2x2
        #out = self.features(x)   # 64x2048
        #print('SHAPE OF x: ', x.shape)
        out = self.features(x)
        #print('SHAPE OF ENCODER OUTPUT: ', out[0].shape)
        return out


# Class of the used network from segmentation models pytorch, derived from a RESNET
class FeatureExtractorChallenge(nn.Module):
    def __init__(self, model):
        super(FeatureExtractorChallenge, self).__init__()
        # Extract RESNET Encoder from the given model
        self.modules = list(model.encoder.children())[:]
        self.features = nn.Sequential(*self.modules)
        self.pooling = nn.MaxPool2d(2, stride=2)
        #self.pooling = nn.AvgPool2d(2, stride=2)
        self.flatten_layer = nn.Flatten()



    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        #print('SHAPE OF x: ', x.shape)
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten_layer(out)
        #print('SHAPE OF ENCODER OUTPUT: ', out[0].shape)
        return out


# It extracts features from a given batch using weights of RESNET
def extract_feature_batch(batch_array):
    batch_array = batch_array.to(device)
    with torch.no_grad():

    # Extract the feature from the image
        feature = new_model(batch_array.float())
        feature = feature.cpu().detach().numpy()   #.reshape(-1)
    return feature


# This function extracts the features from a single image and returns a numpy array with these features
def extract_feature_image(img):
    #start = datetime.now()
    features = []     # Will contain the feature

    size = 64
    stride = 32
    img = torch.from_numpy(img).unsqueeze(0)
    patches = img.unfold(1, size, stride).unfold(2, size, stride)
    patches = torch.reshape(patches, (patches.shape[1]*patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]))


    # Division of the image in patches of 64x64 and stride = 32. for every patch features are computed
    # using RESNET50 and appended in the list features


    batchsize = 64

    for i in range(0, patches.shape[0], batchsize):
        X_batch = patches[i: i + batchsize]
        features_batch = extract_feature_batch(X_batch)
        if (i == 0):
            features = np.copy(features_batch)
        else:
            features = np.concatenate((features, features_batch), axis=0)

    # scaler = MinMaxScaler(feature_range=(-1,1))
    # scaler.fit_transform(features)

    features = features.flatten()

    # power normalization, also called square-rooting normalization
    #features = np.sign(features) * np.sqrt(np.abs(features))
    # L2 normalization
    #features = features / np.sqrt(np.dot(features, features))

    # v_min, v_max = features.min(), features.max()
    # new_min, new_max = -1., 1.
    # features = (features - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
    #print(datetime.now() - start)
    return features


# It computes feature extraction for all images in the list and creates an external npy file with the features array
def compute_features(name_experiment, name_video, list_images, path_images, num_features_image):
    features_array = np.zeros((len(list_images), num_features_image), dtype=float)
    path_dictionary_folder = os.path.join(DATASET_FILES_PATH, 'dictionary_file_npy', name_experiment)
    if not os.path.exists(path_dictionary_folder):
        os.makedirs(path_dictionary_folder)

    path_dictionary_npy = os.path.join(DATASET_FILES_PATH, 'dictionary_' + name_video + '.npy')
    if not os.path.exists(path_dictionary_npy):
        for i in range(features_array.shape[0]):
            #print(i)
            img = cv2.imread(path_images + '/' + list_images[i]+'.png')
            #print(path_images + '/' + list_images[i])
            img = img.astype(float)/255.
            features = extract_feature_image(img)
            features_array[i, :] = features

        np.save(path_dictionary_npy, features_array)

    else:
        features_array = np.load(path_dictionary_npy)

    return features_array

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

# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask):
    path_panorama_image_folder = os.path.join(DATASET_FILES_PATH, 'output_panorama_images',
                                              name_experiment)
    if not os.path.exists(path_panorama_image_folder):
        # print('creating new panorama image folder')
        os.makedirs(path_panorama_image_folder)

    path_panorama_video_folder = os.path.join(DATASET_FILES_PATH, 'output_panorama_video',
                                              name_experiment)
    if not os.path.exists(path_panorama_video_folder):
        # print('creating new panorama video folder')
        os.makedirs(path_panorama_video_folder)

    path_panorama_image = os.path.join(DATASET_FILES_PATH, 'output_panorama_images', name_experiment,
                                       'panorama_' + name_video + '.png')


    if not os.path.exists(path_panorama_image):

        black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
        panorama = np.copy(black_canvas)

        mask = np.uint8(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        #parameters for the video
        size = (black_canvas.shape[1], black_canvas.shape[0])
        fps = 25
        path_video = os.path.join(DATASET_FILES_PATH,'output_panorama_video', name_experiment,name_video + '.mp4')
        out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

        for name in list_images:
            if('anon' not in name):
                list_images.remove(name)


        for index, image_name in enumerate(list_images):
            image = cv2.imread(os.path.join(os.getcwd(),path_images,image_name+'.png'),1)
            print(image_name)

            image_circle = cv2.bitwise_and(image, mask*255)

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
            #plt.imshow(panorama_with_border)
            H4p_prev = H4p
            out.write(panorama_with_border)

        out.release()
        path_panorama = os.path.join(DATASET_FILES_PATH,'output_panorama_images',name_experiment,'panorama_' + name_video + '.png')
        cv2.imwrite(path_panorama, panorama_with_border)
        cv2.destroyAllWindows()

        return panorama_with_border

    else:
        #print('Panorama already reconstructed')
        panorama = cv2.imread(path_panorama_image, 1)
        return panorama



# It computes an array of features of the frames defined as key frames
def compute_key_frames(list_images, list_matrices, mask):

    x = np.arange(len(list_images_common))  # defines the x values of points in the graph
    list_y_points = []  # defines the y values of points in the graph

    black_canvas = np.zeros((CANVAS_SHAPE, CANVAS_SHAPE, 3), dtype="uint8")
    canvas_center = np.array([[np.cos(0), np.sin(0), black_canvas.shape[1] / 2], [-np.sin(0), np.cos(0), black_canvas.shape[0] / 2],[0, 0, 1]])
    img_origin = np.array([[np.cos(0), np.sin(0), -mask.shape[1] / 2], [-np.sin(0), np.cos(0), -mask.shape[0] / 2], [0, 0, 1]])
    initial_matrix = img_origin @ canvas_center
    number_of_white_pix_mask = np.sum(mask == 1)  # extracting only white pixels
    list_key_frames = []   # contains the names of the key frames
    list_indexes_key_frames = []   # contains the list of the indexes of the key frames
    matrices_key_frames = []   # contains the matrices of the key frames referred to the first image in the centre of the canvas


    for count in range(len(list_images)):

        if count == 0:   # the first frame is a key frame

            transformation_matrix = initial_matrix @ list_matrices[count]
            current_matrix_from_origin = transformation_matrix
            mask_current_warped = cv2.warpPerspective(mask, np.float32(transformation_matrix), (black_canvas.shape[1], black_canvas.shape[0]), flags=cv2.INTER_NEAREST)
            current_matrix = transformation_matrix
            actual_key_mask = mask_current_warped
            list_key_frames.append(list_images[count])
            list_indexes_key_frames.append(count)
            matrices_key_frames.append(current_matrix_from_origin)
            y = 100.0
            list_y_points.append(y)


        else:

            matrix_chain_rule = np.matmul(current_matrix, list_matrices[count])
            current_matrix_from_origin = np.matmul(current_matrix_from_origin, list_matrices[count])
            mask_current_warped = cv2.warpPerspective(mask, np.float32(matrix_chain_rule), (black_canvas.shape[1], black_canvas.shape[0]), flags=cv2.INTER_NEAREST)

            overlap = cv2.bitwise_and(mask_current_warped, actual_key_mask)
            num_white_pix_first_term = np.sum(mask_current_warped == 1)
            num_white_pix_second_term = np.sum(actual_key_mask == 1)
            num_white_pix_max = max(num_white_pix_first_term, num_white_pix_second_term)
            #cv2.imwrite(os.path.join(path_overlap,'frame_'+'{:03d}'.format(count)+'.png'), overlap*255)

            number_of_white_pix_overlap = np.sum(overlap == 1)  # extracting only white pixels
            percentage_overlap = number_of_white_pix_overlap * 100 / num_white_pix_max
            list_y_points.append(percentage_overlap)

            #im_h = cv2.hconcat([mask_current_warped, actual_key_mask, overlap])
            #cv2.imwrite(os.path.join(path_overlap,'frame_'+'{:03d}'.format(count)+'.png'), im_h*255)


            if (percentage_overlap < 50): # the frame is the new key frame

                transformation_matrix = initial_matrix
                mask_current_warped = cv2.warpPerspective(mask, np.float32(transformation_matrix), (black_canvas.shape[1], black_canvas.shape[0]),flags=cv2.INTER_NEAREST)
                actual_key_mask = mask_current_warped
                list_key_frames.append(list_images[count])
                list_indexes_key_frames.append(count)
                matrices_key_frames.append(current_matrix_from_origin)
                current_matrix = [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]
                current_matrix = current_matrix @ initial_matrix


            else: # the current frame is too overlapped to be a new key frame
                current_matrix = matrix_chain_rule


    return list_key_frames, matrices_key_frames



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




# It performs a sanity check between the original images and the frame to reattach with an applied transformation
def sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha, ill_beta, rot, crop_red, patch_dim):
    panorama = panorama * 0.5

    #list frames contain the names of the original frames (with no transformation)
    #features_array_buffer contains the features of the images with transformation applied
    image_to_check = cv2.imread(path_images + '/' + frame_to_reattach + '.png', 1)


    # apply to image_to_check a transformation: rotation, illumination change ir scaling
    image_to_check_transf = add_transformation(image_to_check, ill_alpha, ill_beta, rot, crop_red, patch_dim)
    features_image_to_check = extract_feature_image(image_to_check_transf)

    list_differences = []
    for i in range(features_key_frames.shape[0]):
        dist = np.linalg.norm(features_image_to_check - features_key_frames[i, :])
        #dist = np.dot(features_image_to_check, features_key_frames[i, :]) / (np.sqrt(np.dot(features_image_to_check, features_image_to_check)) * np.sqrt(np.dot(features_key_frames[i, :], features_key_frames[i, :])))
        list_differences.append(dist)

    #print(list_differences)
    #print(list_key_frames)

    min_dist = min(list_differences)
    min_index = list_differences.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]
    min_kf_matrix = matrices_key_frames[min_index]

    tuple_result = (min_index, key_frame_min_dist, min_kf_matrix)

    # METRIC CALCULATION

    if (frame_to_reattach[6:] < key_frame_min_dist[6:]):
        for name in list_images_common:
            if name == frame_to_reattach:
                start_index = list_images_common.index(name)
            if name == key_frame_min_dist:
                end_index = list_images_common.index(name)

        relative_matrix = np.eye(3)
        for i in range(start_index, end_index, 1):
            relative_matrix = relative_matrix @ list_matrices_common[i]

        metric_ssim = compute_metric(frame_to_reattach + '.png', tuple_result[1] + '.png', relative_matrix, path_images)
        print(metric_ssim)

    else:
        for name in list_images_common:
            if name == frame_to_reattach:
                end_index = list_images_common.index(name)
            if name == key_frame_min_dist:
                start_index = list_images_common.index(name)

        relative_matrix = np.eye(3)
        for i in range(start_index, end_index, 1):
            relative_matrix = relative_matrix @ list_matrices_common[i]
        metric_ssim = compute_metric(tuple_result[1] + '.png', frame_to_reattach + '.png', relative_matrix, path_images)
        print(metric_ssim)


    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    key_frame = cv2.imread(path_images + '/' + key_frame_min_dist + '.png')
    key_frame_warped = cv2.warpPerspective(key_frame, np.float32(matrices_key_frames[min_index]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_key_frame = cv2.warpPerspective(mask, np.float32(matrices_key_frames[min_index]), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
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
    #cv2.imwrite(os.path.join(os.getcwd(), 'panorama_key_frame.png'), panorama_with_key_frame)


    image_to_check_warped = cv2.warpPerspective(image_to_check_transf, np.float32(matrix_frame_to_reattach), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
    mask_image_to_check = cv2.warpPerspective(mask, np.float32(matrix_frame_to_reattach), (panorama.shape[1], panorama.shape[0]),flags=cv2.INTER_NEAREST)
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

    path_folder = os.path.join(DATASET_FILES_PATH, 'sanity_check_panorama', name_experiment)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    path_folder = os.path.join(DATASET_FILES_PATH, 'sanity_check_panorama', name_experiment, name_video)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    cv2.imwrite(os.path.join(path_folder, 'sc_rot'+str(rot)+'_contr'+str(ill_alpha)+'_bright'+str(ill_beta)+'_crop'+str(crop_red)+'_patch'+str(patch_dim)+'.png'), panorama_with_key_frame)
    return tuple_result, metric_ssim









#MAIN FLOW EXECUTION


# Network parameters initialization for Resnet from torchvision models
#model = models.vgg16_bn(pretrained=True)
# model = models.resnet50(pretrained=True)
# new_model = FeatureExtractor(model)
#
# # Change the device to GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# new_model = new_model.to(device)


model = models.resnet50(pretrained=True)
new_model = FeatureExtractor(model)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

list_videos = sorted(os.listdir(DATASET_PATH))   # contains list of videos names

index_column = 0
index_row = 0

experiment_result = np.zeros((18,len(list_videos)), dtype=object)

for name_video in list_videos:
    print(name_video)
    path_images = os.path.join(DATASET_PATH, name_video,'images')
    path_matrices = os.path.join(DATASET_PATH, name_video,'output_sp_RANSAC')
    mask = cv2.imread(os.path.join(DATASET_PATH, name_video, 'mask.png'), 0)
    mask = mask/255. #binary mask
    shape_image = mask.shape[0]
    number_of_white_pix_mask = np.sum(mask == 1)  # extracting only white pixels
    number_of_black_pix_mask = np.sum(mask == 0)  # extracting only black pixels
    num_pixel_mask = number_of_white_pix_mask + number_of_black_pix_mask

    num_features_image = 346112   # for RESNET50
    #num_features_image = 86528   # for RESNET18 and RESNET34
    #num_features_image = 1384448 # for RESNET50 from challenge without pooling

    list_images = sorted(os.listdir(path_images)) # list containing all the images names
    list_matrices = sorted(os.listdir(path_matrices)) # list containing all the matrices files names

    list_images_no_extension = [] # list containing all the images names without the extension .png
    list_matrices_no_extension = [] # list containing all the matrices files names without the extension .txt

    for name_image in list_images:
        name_image = name_image.replace('.png', '')
        list_images_no_extension.append(name_image)

    for name_matrix in list_matrices:
        name_matrix = name_matrix.replace('.txt', '')
        list_matrices_no_extension.append(name_matrix)

    list_images_no_extension.sort()
    list_matrices_no_extension.sort()

    list_images_common = [] # list containing all the names of images with corresponding matrices
    for name in list_images_no_extension:
        if name in list_matrices_no_extension:
            list_images_common.append(name)


    list_matrices_names_common = [] # list containing all the names of matrices with corresponding images
    for name in list_matrices_no_extension:
        if name in list_images_no_extension:
            list_matrices_names_common.append(name)


    list_images_common.sort()
    list_matrices_names_common.sort()

    list_matrices_common = []
    for count in range(len(list_matrices_names_common)):
        current_matrix = loadtxt(path_matrices + '/' + list_matrices_names_common[count] + '.txt')
        list_matrices_common.append(current_matrix)


    # list_matrices_common contains all the matrices with correspondent image
    # list_images_common contains all the images names with correspondent matrices
    # path_images is the path to the images folder
    # path_matrices is the path to the matrices folder
    panorama = panorama_reconstruction(list_images_common, list_matrices_common, path_images, name_video, mask)

    canvas_center = np.array([[np.cos(0), np.sin(0), panorama.shape[1] / 2], [-np.sin(0), np.cos(0), panorama.shape[0] / 2], [0, 0, 1]])
    img_origin = np.array([[np.cos(0), np.sin(0), -mask.shape[1] / 2], [-np.sin(0), np.cos(0), -mask.shape[0] / 2],[0, 0, 1]])
    first_transformation_matrix = img_origin @ canvas_center



    list_key_frames, matrices_key_frames = compute_key_frames(list_images_common, list_matrices_common, mask)

    build_video_key_frame(list_key_frames, name_video, path_images, mask, list_matrices_common, list_images_common)

    num_frame_to_reattach = 51
    frame_to_reattach = list_images_common[num_frame_to_reattach]

    if (frame_to_reattach in list_key_frames):
        num_frame_to_reattach = 50
        frame_to_reattach = list_images_common[num_frame_to_reattach]

    image_to_reattach = cv2.imread(path_images + '/' + frame_to_reattach+'.png', 1)
    matrix_frame_to_reattach = first_transformation_matrix
    for i in range(num_frame_to_reattach):
        matrix_frame_to_reattach = matrix_frame_to_reattach @ list_matrices_common[i]


    features_key_frames = compute_features(name_experiment, name_video, list_key_frames, path_images, num_features_image)

    list_metric_ssim = []

    # sanity check with no transformation applied
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=0, patch_dim=0)
    experiment_result[0, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with 10 degrees rotation
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=10, crop_red=0, patch_dim=0)
    experiment_result[1, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with 30 degrees rotation
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=30, crop_red=0, patch_dim=0)
    experiment_result[2, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)


    # sanity check with 60 degrees rotation
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=60, crop_red=0, patch_dim=0)
    experiment_result[3, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast alpha=0.8
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=0.80, ill_beta=1.0, rot=0, crop_red=0, patch_dim=0)
    experiment_result[4, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast alpha=0.9
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=0.90, ill_beta=1.0, rot=0, crop_red=0, patch_dim=0)
    experiment_result[5, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast alpha=1.10
    tuple_result = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.10, ill_beta=1.0, rot=0, crop_red=0, patch_dim=0)
    experiment_result[6, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast alpha=1.20
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.20, ill_beta=1.0, rot=0, crop_red=0, patch_dim=0)
    experiment_result[7, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast beta=0.8
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames,list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=0.8, rot=0, crop_red=0, patch_dim=0)
    experiment_result[8, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast beta=0.9
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames,list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=0.9, rot=0, crop_red=0, patch_dim=0)
    experiment_result[9, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast beta=1.10
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames,list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.10, rot=0, crop_red=0, patch_dim=0)
    experiment_result[10, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with contrast beta=1.20
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames,list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.20, rot=0, crop_red=0, patch_dim=0)
    experiment_result[11, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check crop reduction=20
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=20, patch_dim=0)
    experiment_result[12, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check crop reduction=30
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=30, patch_dim=0)
    experiment_result[13, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check crop reduction=50
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=50, patch_dim=0)
    experiment_result[14, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    # sanity check with patch insertion of patch_dim=100
    tuple_result, metric_ssim = sanity_check(name_experiment, frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=0, patch_dim=100)
    experiment_result[15, index_column] = tuple_result
    list_metric_ssim.append(metric_ssim)

    np_metric_ssim = np.array(list_metric_ssim)
    path_folder = os.path.join(os.path.join(DATASET_FILES_PATH, 'boxplots_npy', name_experiment))
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    np.save(os.path.join(path_folder,'ssim_relocalization_' + name_video), np_metric_ssim)


    # code to add the interval of key frames between which we can find the frame to reattach
    list_key_frames_num = [item[8:] for item in list_key_frames]
    frame_to_reattach_num = frame_to_reattach[8:]
    list_tuples_frames_num = []
    for i in range(len(list_key_frames_num)):
        list_tuples_frames_num.append((list_key_frames_num[i],i))

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
    tuple_frame_to_reattach = (frame_to_reattach, interval)

    experiment_result[16, index_column] = tuple_frame_to_reattach
    experiment_result[17, index_column] = list_key_frames

    index_column = index_column +1



#transform array experiment result in an external result sheet

#convert array to pandas dataframe
#print(experiment_result)
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

filepath = os.path.join(path_folder_experiments, name_experiment+'.xlsx')
df.to_excel(filepath, index=True)



