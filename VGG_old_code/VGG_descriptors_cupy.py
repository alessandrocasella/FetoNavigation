import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torchvision import models, transforms
model = models.vgg16(pretrained=True)
from tqdm import tqdm
import numpy as np
import cupy as cp
import cv2
import imutils


# this function extracts the features from a single image and returns a numpy array with these features
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
    features = cp.array(features)
    return features
    #print(features)
    #print(features.shape)

# it computes feature extraction for all images in the buffer and creates an external npy file with the features array
def compute_features_original(features_array_buffer, name_video):
    path_dictionary_npy = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/dictionary_file_npy', 'dictionary_original_' + name_video + '.npy')
    if not os.path.exists(path_dictionary_npy):
        for i in range(50):
            print(i)
            img = cv2.imread(path_folder_frames + '/' + list_frames[i])
            features = extract_feature_image(img)
            features_array_buffer[i, :] = features

        np.save(path_dictionary_npy, features_array_buffer)

    else:
        features_array_buffer = np.load(path_dictionary_npy)
    print(features_array_buffer.shape)
    return features_array_buffer


# it performs a sanity check on original images: applies L2 between the given frames and all frames and computes
# ssim between the given frames and all the others
def sanity_check_original(list_frames, features_array_buffer, path_folder_frames):
    frame_to_check = list_frames[25]
    row_frame_to_check = features_array_buffer[25, :]

    list_differences = []
    for i in range(50):
        image_reference = cv2.imread(path_folder_frames + '/' + frame_to_check)
        image_reference = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)

        image_to_check = cv2.imread(path_folder_frames + '/' + list_frames[i])
        image_to_check = cv2.cvtColor(image_to_check, cv2.COLOR_BGR2GRAY)

        im1 = cv2.GaussianBlur(image_reference, (9, 9), cv2.BORDER_DEFAULT)
        im2 = cv2.GaussianBlur(image_to_check, (9, 9), cv2.BORDER_DEFAULT)
        ssim_index = ssim(im1, im2)

        dist = np.linalg.norm(row_frame_to_check - features_array_buffer[i, :])
        # print(dist)
        list_differences.append((dist, ssim_index))

    # print(list_differences)
    for i in range(len(list_differences)):
        if (list_differences[i][0] == 0.0):
            print('The frame detected is: ', i)
            print('The ssim for this pair is : ', list_differences[i][1])


# it computes feature extraction for all images in the buffer and creates an external npy file with the features array
#Possible transformations applied: rotation
def compute_features_transformation(features_array_buffer, name_video):
    # path_dictionary_npy = os.path.join(os.getcwd(), 'dictionary_file_npy', 'dictionary_rotation5_' + name_video + '.npy')
    # if not os.path.exists(path_dictionary_npy):
    for i in range(50):
        print(i)
        img = cv2.imread(path_folder_frames + '/' + list_frames[i])

        #print(img.dtype)
        #application of a rotation to test robustness of the features
        # img = rotate(img, angle=0)
        # img = img.astype(np.uint8)
        if (i ==25):
            img = imutils.rotate(img, 5)
        #print(img.dtype)

        features = extract_feature_image(img)
        print(features.shape)
        print(type(features))
        print(features_array_buffer.shape)
        print(type(features_array_buffer))
        print(features_array_buffer[i, :].shape)
        print(type(features_array_buffer[i, :]))
        features_array_buffer[i, :] = features

    # np.save(path_dictionary_npy, features_array_buffer)

    # else:
    #     features_array_buffer = np.load(path_dictionary_npy)

    return features_array_buffer



# it performs a sanity check on transformed images: applies L2 between the given frames and all frames and computes
# ssim between the given frames and all the others
def sanity_check_transformation(list_frames, features_array_buffer, path_folder_frames):
    #list frames contain the names of the original frames (with no transformation)
    #features_array_buffer contains the features of the images with transformation applied
    frame_to_check = list_frames[25]
    image_reference = cv2.imread(path_folder_frames + '/' + frame_to_check)
    #image_reference = image_reference.astype(np.uint8)
    row_frame_to_check = extract_feature_image(image_reference)
    print(row_frame_to_check.shape)
    image_reference = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)

    list_differences = []
    for i in range(50):

        image_to_check = cv2.imread(path_folder_frames + '/' + list_frames[i])
        image_to_check = cv2.cvtColor(image_to_check, cv2.COLOR_BGR2GRAY)
        image_reference = image_reference.astype(float)

        #apply to image_to_check the same transformation applied during feature extraction: rotation
        #image_to_check = rotate(image_to_check, angle=0)
        if (i==25):
            image_to_check = imutils.rotate(image_to_check, 5)
        image_to_check = image_to_check.astype(float)

        im1 = cv2.GaussianBlur(image_reference, (5, 5), cv2.BORDER_DEFAULT)
        im2 = cv2.GaussianBlur(image_to_check, (5, 5), cv2.BORDER_DEFAULT)
        #print(im1.shape)
        #print(im2.shape)
        ssim_index = ssim(im1, im2)

        dist = cp.linalg.norm(row_frame_to_check - features_array_buffer[i, :])
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

    print('ROTATION OF 5 DEGREES')
    print('the frame with minimum distance from frame 25 is: ', frame_min_dist)
    print('the euclidean distance is: ', min_dist)
    print('the euclidean distance of frame 25 is: ', list_differences[25][0])
    print('the ssim is: ', list_differences[25][1])
    #print(list_differences)





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

path_folder_frames = os.path.join(os.getcwd(), 'ipcai_superpoint_ransac/anon001/images')
name_video = 'anon001'
list_frames = os.listdir(path_folder_frames)
#print(list_frames)

features_array_buffer = cp.zeros((50,4096), dtype=float)
#features_array_buffer = cp.array(features_array_buffer)

#features_array_buffer = compute_features_original(features_array_buffer, name_video)
features_array_buffer = compute_features_transformation(features_array_buffer, name_video)
sanity_check_transformation(list_frames, features_array_buffer, path_folder_frames)
