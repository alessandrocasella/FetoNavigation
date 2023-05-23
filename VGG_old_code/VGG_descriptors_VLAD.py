from sklearn.decomposition import PCA
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import imutils
from sklearn.cluster import KMeans


# DEFINE GLOBAL PARAMETERS TO APPLY TRANSFORMATIONS TO THE IMAGE AND PERFORM SANITY CHECK
global_illumination_alpha = 1.0    # Simple contrast control
global_illumination_beta = 1.0   # Simple brightness control
global_rotation = 0   # rotation angle in degrees
global_crop = 460   #crop used to perform rescaling


# It calculates a visual dictionary from a set of descriptors
# training = a set of descriptors
def  kMeansDictionary(training, k):

    #K-means algorithm
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
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


def improvedVLAD(descriptor_array, visual_dictionary):
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

def compute_visual_dictionary(features_array):
    visualDictionary=kMeansDictionary(features_array,5)
    return visualDictionary



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
    image = cv2.resize(image, (global_crop, global_crop))
    image = cv2.resize(image, (shape, shape))

    return image


# This function extracts the features from a single image and returns a numpy array with these features
def extract_feature_image(img):
    # Will contain the feature
    features = []

    # Transform the image, so it becomes readable with the model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor()
    ])


    img = transform(img)
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    with torch.no_grad():
    # Extract the feature from the image
        feature = new_model(img)
    # Convert to NumPy Array, Reshape it, and save it to features variable
        features.append(feature.cpu().detach().numpy().reshape(-1))

    # Convert to NumPy Array
    features = np.array(features)

    return features



# It computes feature extraction for all images in the buffer and creates an external npy file with the features array
def compute_features_original(list_frames, features_array, name_video):
    path_dictionary_npy = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/dictionary_file_npy/VLAD', 'dictionary_original_all_video_' + name_video + '.npy')
    if not os.path.exists(path_dictionary_npy):
        for i in range(len(list_frames)):
            print(i)
            img = cv2.imread(path_folder_frames + '/' + list_frames[i])
            features = extract_feature_image(img)
            features_array[i, :] = features

        np.save(path_dictionary_npy, features_array)

    else:
        features_array = np.load(path_dictionary_npy)

    print(features_array.shape)
    #apply VLAD method
    visual_dictionary = compute_visual_dictionary(features_array)
    vlad_vector = improvedVLAD(features_array, visual_dictionary)
    print(vlad_vector.shape)

    return features_array


# It performs a sanity check on original images: applies L2 between the given frames and all frames and computes
# ssim between the given frames and all the others
def sanity_check_original(list_frames, features_array, path_folder_frames):
    frame_to_check = list_frames[25]
    row_frame_to_check = features_array[25, :]

    list_differences = []
    for i in range(len(list_frames)):
        image_reference = cv2.imread(path_folder_frames + '/' + frame_to_check)
        image_reference = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)

        image_to_check = cv2.imread(path_folder_frames + '/' + list_frames[i])
        image_to_check = cv2.cvtColor(image_to_check, cv2.COLOR_BGR2GRAY)

        im1 = cv2.GaussianBlur(image_reference, (9, 9), cv2.BORDER_DEFAULT)
        im2 = cv2.GaussianBlur(image_to_check, (9, 9), cv2.BORDER_DEFAULT)
        ssim_index = ssim(im1, im2)

        dist = np.linalg.norm(row_frame_to_check - features_array[i, :])
        # print(dist)
        list_differences.append((dist, ssim_index))

    # print(list_differences)
    for i in range(len(list_differences)):
        if (list_differences[i][0] == 0.0):
            print('The frame detected is: ', i)
            print('The ssim for this pair is : ', list_differences[i][1])


# It computes feature extraction for all images in the buffer and creates an external npy file with the features array
def compute_features_transformation(list_frames, features_array, name_video):
    path_dictionary_npy = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/dictionary_file_npy', 'dictionary_sca_460_all_video_' + name_video + '.npy')
    if not os.path.exists(path_dictionary_npy):
        for i in range(len(list_frames)):
            print(i)
            img = cv2.imread(path_folder_frames + '/' + list_frames[i])

            #print(img.dtype)
            #application of a rotation to test robustness of the features
            # img = rotate(img, angle=0)
            # img = img.astype(np.uint8)
            if (i ==25):
                img = add_transformation(img)
            #print(img.dtype)

            features = extract_feature_image(img)
            features_array[i, :] = features

        np.save(path_dictionary_npy, features_array)

    else:
        features_array = np.load(path_dictionary_npy)

    img = cv2.imread(path_folder_frames + '/' + list_frames[25])
    # features_array_to_check contains the original images. By applying PCA also on this array,
    # it it possible to use it to perform sanity check between the original image and the transformed one
    features_array_to_check = np.copy(features_array)
    features_array_to_check[25, :] = extract_feature_image(img)

    # Apply PCA on the array of features of transformed images
    pca = PCA(n_components=256)
    features_array= pca.fit_transform(features_array)
    print(features_array.shape)

    # Apply PCA on the array of features of original images
    pca = PCA(n_components=256)
    features_array_to_check = pca.fit_transform(features_array_to_check)
    print(features_array_to_check.shape)

    return features_array, features_array_to_check



# It performs a sanity check on transformed images: applies L2 between the given frames and all frames and computes
# ssim between the given frames and all the others
def sanity_check_transformation(list_frames, features_array, path_folder_frames, features_array_to_check):
    #list frames contain the names of the original frames (with no transformation)
    #features_array_buffer contains the features of the images with transformation applied
    frame_to_check = list_frames[25]
    image_reference = cv2.imread(path_folder_frames + '/' + frame_to_check)
    #image_reference = image_reference.astype(np.uint8)
    row_frame_to_check = features_array_to_check[25, :]
    #print(row_frame_to_check.shape)
    image_reference = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)

    list_differences = []
    for i in range(len(list_frames)):

        image_to_check = cv2.imread(path_folder_frames + '/' + list_frames[i])
        image_to_check = cv2.cvtColor(image_to_check, cv2.COLOR_BGR2GRAY)
        image_reference = image_reference.astype(float)

        #apply to image_to_check the same transformation applied during feature extraction: rotation
        #image_to_check = rotate(image_to_check, angle=0)
        if (i==25):
            image_to_check = add_transformation(image_to_check)
        image_to_check = image_to_check.astype(float)

        im1 = cv2.GaussianBlur(image_reference, (5, 5), cv2.BORDER_DEFAULT)
        im2 = cv2.GaussianBlur(image_to_check, (5, 5), cv2.BORDER_DEFAULT)
        #print(im1.shape)
        #print(im2.shape)
        ssim_index = ssim(im1, im2)

        dist = np.linalg.norm(row_frame_to_check - features_array[i, :])
        # print(dist)
        list_differences.append((dist, ssim_index))

    min_dist = 10000
    frame_min_dist = 0
    # print(list_differences)
    for i in range(len(list_differences)):
        # if (list_differences[i][0] == 0.0):
        #     ssim_pair = list_differences[i][1]
        #     print('The frame detected is: ', i)
        #     print('The ssim for this pair is : ', ssim_pair)
        # elif (i==25):
        #     print('the euclidean distance for the frame 25 is: ',list_differences[25][0])
        #     print('the ssim for the frame 25 is: ', list_differences[25][1])
        if (list_differences[i][0]) < min_dist:
            min_dist = list_differences[i][0]
            frame_min_dist = i

    print('WITH PCA')
    print('ROTATION OF', global_rotation, 'degrees')
    print('CHANGE ILLUMINATION: Contrast is', global_illumination_alpha, 'brightness is', global_illumination_beta)
    print('RESCALING USING A CROP OF', global_crop)
    print('the frame with minimum distance from frame 25 is: ', frame_min_dist)
    print('the euclidean distance is: ', min_dist)
    print('the euclidean distance of frame 25 is: ', list_differences[25][0])
    print('the ssim is: ', list_differences[25][1])





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




# MAIN FLOW OF EXECUTION

model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

path_folder_frames = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset/anon001/images')
name_video = 'anon001'
list_frames = os.listdir(path_folder_frames)
length_list = len(list_frames)
#print(list_frames)

features_array = np.zeros((length_list,4096), dtype=float)

# These functions perform the feature extraction and the sanity check with the original images
features_array = compute_features_original(list_frames, features_array, name_video)
sanity_check_original(list_frames, features_array, path_folder_frames)


# These functions perform the feature extraction and the sanity check between a frame not transformed
# and the same frame with a transformation applied. Possible transformations: rotation
#features_array, features_array_to_check = compute_features_transformation(list_frames, features_array, name_video)
#sanity_check_transformation(list_frames, features_array, path_folder_frames, features_array_to_check)
