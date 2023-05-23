import imutils
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import loadtxt
import os
from sklearn.cluster import KMeans
from torch.nn.functional import interpolate
import torch
from torch import optim, nn
from torchvision import models, transforms
from datetime import datetime


# DEFINE GLOBAL PARAMETERS TO APPLY TRANSFORMATIONS TO THE IMAGE AND PERFORM SANITY CHECK
global_illumination_alpha = 1.0    # Simple contrast control
global_illumination_beta = 1.0   # Simple brightness control
global_rotation = 0   # rotation angle in degrees
global_crop = 720   #crop used to perform rescaling


# It calculates a visual dictionary from a set of descriptors
# training = a set of descriptors
def kMeansDictionary(training, k):

    #K-means algorithm
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=0).fit(training)
    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
    return est
    #clf2 = pickle.loads(s)


# It applies VLAD to the descriptor array in input given a visual dictionary
def VLAD(descriptor_array, visual_dictionary):
    predictedLabels = visual_dictionary.predict(descriptor_array)
    centers = visual_dictionary.cluster_centers_
    labels = visual_dictionary.labels_
    k = visual_dictionary.n_clusters

    m, d = descriptor_array.shape
    V = np.zeros([k, d])
    # computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels == i) > 0:
            # add the diferences
            V[i] = np.sum(descriptor_array[predictedLabels == i, :] - centers[i], axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V) * np.sqrt(np.abs(V))

    # L2 normalization

    V = V / np.sqrt(np.dot(V, V))
    return V


# It computes the visual dictionary necessary to compute VLAD
def compute_visual_dictionary(features_array):
    visualDictionary=kMeansDictionary(features_array,5)
    return visualDictionary

# It extracts features from a given batch using weights of VGG16
def extract_feature_batch(batch_array):
    resize_batch = interpolate(batch_array,(448,448))
    #resize_batch = batch_array.Resize((448,448), interpolation = 'bilinear')
    #img = img.reshape(1, 3, 448, 448)
    resize_batch = resize_batch.to(device)
    with torch.no_grad():
    # Extract the feature from the image
        feature = new_model(resize_batch.float())
        feature = feature.cpu().detach().numpy()   #.reshape(-1)
        #print(feature.shape)
    return feature


# This function extracts the features from a single image and returns a numpy array with these features
def extract_feature_image(img):

    # Will contain the feature
    features = []
    #patches = patchify(img, (64, 64, 3), step=32)
    size = 64
    stride = 32
    img = torch.from_numpy(img).unsqueeze(0)
    #print(img.shape)

    patches = img.unfold(1, size, stride).unfold(2, size, stride)
    #print(patches.shape)
    patches = torch.reshape(patches, (patches.shape[1]*patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]))
    #print('shape of patches: ', patches.shape)


    # Division of the image in patches of 64x64 and stride = 32. for every patch features are computed
    # using VGG16 and appended in the list features
    #patches = np.reshape(patches, (patches.shape[0]*patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]))

    #print(patches.shape)
    batchsize = 64

    for i in range(0, patches.shape[0], batchsize):
        #print('Batch from ',i, 'to ', i + batchsize)
        X_batch = patches[i: i + batchsize]
        #print(X_batch.shape)
        features_batch = extract_feature_batch(X_batch)
        if (i == 0):
            features = np.copy(features_batch)
        else:
            features = np.concatenate((features, features_batch), axis=0)

    #apply VLAD method
    visual_dictionary = compute_visual_dictionary(features)
    vlad_vector = VLAD(features, visual_dictionary)
    #print(vlad_vector.shape)
    vlad_vector = np.array(vlad_vector)

    return vlad_vector


# It computes feature extraction for all images in the buffer and creates an external npy file with the features array
def compute_features(name_video, list_images, path_images):
    features_array = np.zeros((len(list_images_common), 20480), dtype=float)
    path_dictionary_npy = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/dictionary_file_npy/Batch+Patch+VLAD', 'dictionary_original_' + name_video + '.npy')
    if not os.path.exists(path_dictionary_npy):
        for i in range(features_array.shape[0]):
            print(i)
            img = cv2.imread(path_images + '/' + list_images[i]+'.png')
            #print(path_images + '/' + list_images[i])
            img = img.astype(float)/255.
            features = extract_feature_image(img)
            features_array[i, :] = features

        np.save(path_dictionary_npy, features_array)

    else:
        features_array = np.load(path_dictionary_npy)

    return features_array


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    # Initialize the model


# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask):
    black_canvas = np.zeros((2000, 2000, 3), dtype="uint8")
    panorama = np.copy(black_canvas)

    image = cv2.imread(os.path.join(os.getcwd(), path_images, list_images[0] + '.png'), 1)

    mask = np.uint8(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    #parameters for the video
    size = (black_canvas.shape[1], black_canvas.shape[0])
    fps = 25
    path_video = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_video', name_video + '.mp4')
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
    path_panorama = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/output_panorama_images', 'panorama_' + name_video + '.png')
    cv2.imwrite(path_panorama, panorama_with_border)
    cv2.destroyAllWindows()
    # plt.imshow(panorama_with_border)
    # plt.show()
    #cv2.imwrite('panorama.png', panorama_with_border)
    return panorama_with_border


# It computes an array of features of the frames defined as key frames
def compute_features_key_frames(features_array, list_images, list_matrices, mask):

    black_canvas = np.zeros((2000, 2000, 3), dtype="uint8")
    canvas_center = np.array([[np.cos(0), np.sin(0), black_canvas.shape[1] / 2], [-np.sin(0), np.cos(0), black_canvas.shape[0] / 2],[0, 0, 1]])
    img_origin = np.array([[np.cos(0), np.sin(0), -mask.shape[1] / 2], [-np.sin(0), np.cos(0), -mask.shape[0] / 2], [0, 0, 1]])
    initial_matrix = img_origin @ canvas_center
    number_of_white_pix_mask = np.sum(mask == 1)  # extracting only white pixels

    list_key_frames = []
    for count in range(len(list_images)):

        if count == 0: # the first frame is a key frame
            transformation_matrix = initial_matrix @ list_matrices[count]
            mask_current_warped = cv2.warpPerspective(mask, np.float32(transformation_matrix), (black_canvas.shape[1], black_canvas.shape[0]), flags=cv2.INTER_NEAREST)
            # plt.imshow(mask_current_warped)
            # plt.show()
            current_matrix = transformation_matrix
            #actual_key_frame = cv2.imread(path_images + '/' + list_images_common[count] + '.png')
            actual_key_mask = mask_current_warped
            #index_actual_key_frame = count
            list_key_frames.append(list_images[count])
            features_key_frames = features_array[count, :]
            features_key_frames = np.reshape(features_key_frames, (1, 20480))

        else:
            matrix_chain_rule = np.matmul(current_matrix, list_matrices[count])
            mask_current_warped = cv2.warpPerspective(mask, np.float32(matrix_chain_rule), (black_canvas.shape[1], black_canvas.shape[0]), flags=cv2.INTER_NEAREST)

            overlap = cv2.bitwise_and(mask_current_warped, actual_key_mask)
            number_of_white_pix_overlap = np.sum(overlap == 1)  # extracting only white pixels
            percentage_overlap = number_of_white_pix_overlap * 100 / number_of_white_pix_mask
            #print('PERCENTAGE OF OVERLAP: ', percentage_overlap)


            if (percentage_overlap < 50): # the frame is the new key frame
                #actual_key_frame = cv2.imread(path_images + '/' + list_images_common[count] + '.png')
                actual_key_mask = mask_current_warped
                #index_actual_key_frame = count
                list_key_frames.append(list_images[count])
                current_matrix = [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]
                current_matrix = current_matrix @ initial_matrix
                features_array_row = np.reshape(features_array[count, :], (1, 20480))
                features_key_frames = np.concatenate((features_key_frames, features_array_row), axis=0)

            else: # the current frame is too overlapped to be a new key frame
                current_matrix = matrix_chain_rule

    print(list_key_frames)
    print('LEN LIST KEY FRAMES: ', len(list_key_frames))
    return features_key_frames, list_key_frames



# It performs a sanity check between the frame to attach and the original images
def sanity_check_original(frame_to_reattach, features_frame_to_reattach, panorama, features_key_frames, list_key_frames, list_images_common, list_matrices_common):
    list_differences = []
    for i in range(features_key_frames.shape[0]):
        dist = np.linalg.norm(features_frame_to_reattach - features_key_frames[i, :])
        # print(dist)
        list_differences.append(dist)
    #print(list_differences)

    min_dist = min(list_differences)
    min_index = list_differences.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]

    print('Frame to reattach is: ', frame_to_reattach)
    print('Key frame at minimum distance is: ', key_frame_min_dist)
    print('Minimum distance is: ', min_dist)
    print('Index of key frame at minimum distance is: ', min_index)


# It adds a transformation to an image to perform sanity checks and returns the transformed image
# with a string indicating the applied transformation
# Possible transformation: rotation, illumination change (contrast and brightness change)
def add_transformation(image):

    # line to apply rotation
    image = imutils.rotate(image, global_rotation)

    # line to apply illumination changes
    image = cv2.convertScaleAbs(image, alpha=global_illumination_alpha, beta=global_illumination_beta)

    #line to apply scaling
    shape = image.shape[0]
    #image = cv2.resize(image, (global_crop, global_crop))
    #image = cv2.resize(image, (shape, shape))

    return image


# It performs a sanity check between the original images and the frame to reattach with an applied transformation
def sanity_check_transformation(frame_to_reattach, panorama, features_key_frames, list_key_frames, list_images_common, list_matrices_common, path_images):
    #list frames contain the names of the original frames (with no transformation)
    #features_array_buffer contains the features of the images with transformation applied
    image_to_check = cv2.imread(path_images + '/' + frame_to_reattach + '.png')

    # apply to image_to_check a transformation: rotation, illumination change ir scaling
    image_to_check = add_transformation(image_to_check)
    image_to_check = image_to_check.astype(float)
    start = datetime.now()
    features_image_to_check = extract_feature_image(image_to_check)
    print(datetime.now() - start)


    list_differences = []
    for i in range(features_key_frames.shape[0]):
        dist = np.linalg.norm(features_image_to_check - features_key_frames[i, :])
        list_differences.append(dist)

    min_dist = min(list_differences)
    min_index = list_differences.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]

    print('ROTATION OF', global_rotation, 'degrees')
    print('CHANGE ILLUMINATION: Contrast is', global_illumination_alpha, 'brightness is', global_illumination_beta)
    print('RESCALING USING A CROP OF', global_crop)
    print('Frame to reattach is: ', frame_to_reattach)
    print('Key frame at minimum distance is: ', key_frame_min_dist)
    print('Minimum distance is: ', min_dist)
    print('Index of key frame at minimum distance is: ', min_index)



#MAIN FLOW EXECUTION

#notwork parameters initialization
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

name_video = 'anon010'
path_images = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images')
path_matrices = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'output_sp_RANSAC')
mask = cv2.imread(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'mask.png'), 0)
print(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'mask.png'))
mask = mask/255. #binary mask
number_of_white_pix_mask = np.sum(mask == 1.0)  # extracting only white pixels
number_of_black_pix_mask = np.sum(mask == 0.0)  # extracting only black pixels
num_pixel_mask = number_of_white_pix_mask + number_of_black_pix_mask

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
#print(len(list_matrices_common))

# list_matrices_common contains all the matrices with correspondent image
# list_images_common contains all the images names with correspondent matrices
# path_images is the path to the images folder
# path_matrices is the path to the matrices folder
panorama = panorama_reconstruction(list_images_common, list_matrices_common, path_images, name_video, mask)

# Computation of the array of features for the images
features_array = compute_features(name_video, list_images_common, path_images)


frame_to_reattach = list_images_common[50]
features_frame_to_reattach = features_array[50, :]
features_key_frames,list_key_frames = compute_features_key_frames(features_array, list_images_common, list_matrices_common, mask)
sanity_check_original(frame_to_reattach, features_frame_to_reattach, panorama, features_key_frames, list_key_frames, list_images_common, list_matrices_common)
#sanity_check_transformation(frame_to_reattach, panorama, features_key_frames, list_key_frames, list_images_common, list_matrices_common, path_images)

# print(datetime.now()-start)