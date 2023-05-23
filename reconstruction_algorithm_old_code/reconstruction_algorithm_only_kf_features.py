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
from torch.nn.functional import interpolate
import torch
from torch import optim, nn
from torchvision import models, transforms
from matplotlib.animation import FuncAnimation

# Class of the used network, derived from a VGG16
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
        out = self.features(x) # 64x512x14x14
        print('shape after self.features: ', out.shape)
        out = self.pooling(out) # 64x512x7x7
        print('shape after pooling: ', out.shape)
        out = self.flatten(out) # 64x25088
        print('shape after flatten: ', out.shape)
        out = self.fc(out) # 64x4096
        print('shape after fc: ', out.shape)
        return out

    # Initialize the model


# It calculates a visual dictionary from a set of descriptors
# training = a set of descriptors
def kMeansDictionary(training, k):

    #K-means algorithm
    est = KMeans(n_clusters=k, init='k-means++', random_state=3, tol=0.0001, verbose=0).fit(training)
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
    global k_kmeans
    visualDictionary=kMeansDictionary(features_array, k_kmeans)
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
    patches = img.unfold(1, size, stride).unfold(2, size, stride)
    patches = torch.reshape(patches, (patches.shape[1]*patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]))


    # Division of the image in patches of 64x64 and stride = 32. for every patch features are computed
    # using VGG16 and appended in the list features
    #patches = np.reshape(patches, (patches.shape[0]*patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]))

    batchsize = 64

    for i in range(0, patches.shape[0], batchsize):
        X_batch = patches[i: i + batchsize]
        features_batch = extract_feature_batch(X_batch)
        if (i == 0):
            features = np.copy(features_batch)
        else:
            features = np.concatenate((features, features_batch), axis=0)

    #apply VLAD method
    visual_dictionary = compute_visual_dictionary(features)
    vlad_vector = VLAD(features, visual_dictionary)
    vlad_vector = np.array(vlad_vector)
    return vlad_vector


# It computes feature extraction for all images in the list and creates an external npy file with the features array
def compute_features(name_video, list_images, path_images, num_features_image):
    features_array = np.zeros((len(list_images), num_features_image), dtype=float)
    path_dictionary_npy = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/dictionary_file_npy/Batch+Patch+VLAD_only_kf_features/deterministic', 'dictionary_' + name_video + '.npy')
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


# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask):
    path_panorama_image = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/output_panorama_images/panorama_' + name_video + '.png')
    if not os.path.exists(path_panorama_image):

        black_canvas = np.zeros((2000, 2000, 3), dtype="uint8")
        panorama = np.copy(black_canvas)

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

        return panorama_with_border

    else:
        print('Panorama already reconstructed')
        panorama = cv2.imread(path_panorama_image, 1)
        return panorama



# Fuction that sets the data of the animation
def animate(i, x, y, graph):
    if i < len(list_images_common):
        graph.set_data(x[:i+1], y[:i+1])
    else:
        pass
    return graph

# It computes an array of features of the frames defined as key frames
def compute_key_frames(list_images, list_matrices, mask):

    x = np.arange(len(list_images_common))  # defines the x values of points in the graph
    list_y_points = []  # defines the y values of points in the graph

    black_canvas = np.zeros((2000, 2000, 3), dtype="uint8")
    canvas_center = np.array([[np.cos(0), np.sin(0), black_canvas.shape[1] / 2], [-np.sin(0), np.cos(0), black_canvas.shape[0] / 2],[0, 0, 1]])
    img_origin = np.array([[np.cos(0), np.sin(0), -mask.shape[1] / 2], [-np.sin(0), np.cos(0), -mask.shape[0] / 2], [0, 0, 1]])
    initial_matrix = img_origin @ canvas_center
    number_of_white_pix_mask = np.sum(mask == 1)  # extracting only white pixels
    list_key_frames = []   # contains the names of the key frames
    matrices_key_frames = []   # contains the matrices of the key frames referred to the first image in the centre of the canvas

    # path_overlap = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/overlap/'+'overlap_'+name_video)
    # if not os.path.exists(path_overlap):
    #     os.makedirs(path_overlap)

    for count in range(len(list_images)):

        if count == 0: # the first frame is a key frame

            transformation_matrix = initial_matrix @ list_matrices[count]
            current_matrix_from_origin = transformation_matrix
            mask_current_warped = cv2.warpPerspective(mask, np.float32(transformation_matrix), (black_canvas.shape[1], black_canvas.shape[0]), flags=cv2.INTER_NEAREST)
            current_matrix = transformation_matrix
            actual_key_mask = mask_current_warped
            list_key_frames.append(list_images[count])
            matrices_key_frames.append(current_matrix_from_origin)
            y = 100.0
            list_y_points.append(y)

        else:

            matrix_chain_rule = np.matmul(current_matrix, list_matrices[count])
            current_matrix_from_origin = np.matmul(current_matrix_from_origin, list_matrices[count])
            mask_current_warped = cv2.warpPerspective(mask, np.float32(matrix_chain_rule), (black_canvas.shape[1], black_canvas.shape[0]), flags=cv2.INTER_NEAREST)

            overlap = cv2.bitwise_and(mask_current_warped, actual_key_mask)
            #cv2.imwrite(os.path.join(path_overlap,'frame_'+'{:03d}'.format(count)+'.png'), overlap*255)

            number_of_white_pix_overlap = np.sum(overlap == 1)  # extracting only white pixels
            percentage_overlap = number_of_white_pix_overlap * 100 / number_of_white_pix_mask
            list_y_points.append(percentage_overlap)


            if (percentage_overlap < 50): # the frame is the new key frame

                actual_key_mask = mask_current_warped
                list_key_frames.append(list_images[count])
                matrices_key_frames.append(current_matrix_from_origin)
                current_matrix = [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]
                current_matrix = current_matrix @ initial_matrix


            else: # the current frame is too overlapped to be a new key frame
                current_matrix = matrix_chain_rule

    y = np.array(list_y_points)
    fig = plt.figure()
    plt.xlim(0, len(list_images))
    plt.ylim(0, 100.0)
    graph, = plt.plot([], [])

    ani = FuncAnimation(fig, animate, frames=len(list_images), fargs=[x, y, graph], interval=50, repeat=False)
    # plt.show()
    # path_animation = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/animations/animation_overlap/graph_overlap_' + name_video + '.mp4')
    # ani.save(path_animation)

    return list_key_frames, matrices_key_frames

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
    image = cv2.resize(image, (global_crop, global_crop))
    image = cv2.resize(image, (shape, shape))

    return image


# It performs a sanity check between the original images and the frame to reattach with an applied transformation
def sanity_check_transformation(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask):
    #list frames contain the names of the original frames (with no transformation)
    #features_array_buffer contains the features of the images with transformation applied
    image_to_check = cv2.imread(path_images + '/' + frame_to_reattach + '.png', 1)


    # apply to image_to_check a transformation: rotation, illumination change ir scaling
    image_to_check_transf = add_transformation(image_to_check)
    #image_to_check_transf = np.copy(image_to_check)
    #image_to_check_transf = image_to_check_transf.astype(float)
    features_image_to_check = extract_feature_image(image_to_check_transf)
    print(features_image_to_check.shape)
    print(features_key_frames.shape)

    list_differences = []
    for i in range(features_key_frames.shape[0]):
        dist = np.linalg.norm(features_image_to_check - features_key_frames[i, :])
        list_differences.append(dist)

    #print(list_differences)
    print(list_key_frames)

    min_dist = min(list_differences)
    min_index = list_differences.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]

    print('ROTATION OF', global_rotation, 'degrees')
    print('CHANGE ILLUMINATION: Contrast is', global_illumination_alpha, 'brightness is', global_illumination_beta)
    print('RESCALING USING A CROP REDUCTION OF', global_crop_reduction)
    print('Frame to reattach is: ', frame_to_reattach)
    print('Key frame at minimum distance is: ', key_frame_min_dist)
    print('Minimum distance is: ', min_dist)
    print('Index of key frame at minimum distance is: ', min_index)


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


    #panorama_with_frame_to_reattach = np.copy(panorama)
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
    cv2.imwrite(os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/sanity_check_panorama', name_video, 'sc_rot' + str(global_rotation) + '_ill' + str(global_illumination_alpha) + '_crop' + str(global_crop_reduction) + '.png'), panorama_with_key_frame)










#MAIN FLOW EXECUTION

#notwork parameters initialization
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

name_video = 'anon012'
path_images = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images')
path_matrices = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'output_sp_RANSAC')
mask = cv2.imread(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'mask.png'), 0)
mask = mask/255. #binary mask
shape_image = mask.shape[0]
number_of_white_pix_mask = np.sum(mask == 1)  # extracting only white pixels
number_of_black_pix_mask = np.sum(mask == 0)  # extracting only black pixels
num_pixel_mask = number_of_white_pix_mask + number_of_black_pix_mask



# DEFINE GLOBAL PARAMETERS TO APPLY TRANSFORMATIONS TO THE IMAGE AND PERFORM SANITY CHECK
global_illumination_alpha = 1.00    # Simple contrast control
global_illumination_beta = 1.00   # Simple brightness control
global_rotation = 0   # rotation angle in degrees
global_crop_reduction = 20
global_crop = shape_image - global_crop_reduction   #crop used to perform rescaling
k_kmeans = 128
num_features_image = 524288


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

canvas_center = np.array([[np.cos(0), np.sin(0), panorama.shape[1] / 2], [-np.sin(0), np.cos(0), panorama.shape[0] / 2], [0, 0, 1]])
img_origin = np.array([[np.cos(0), np.sin(0), -mask.shape[1] / 2], [-np.sin(0), np.cos(0), -mask.shape[0] / 2],[0, 0, 1]])
first_transformation_matrix = img_origin @ canvas_center

num_frame_to_reattach = 50
frame_to_reattach = list_images_common[num_frame_to_reattach]
image_to_reattach = cv2.imread(path_images + '/' + frame_to_reattach+'.png', 1)
#features_frame_to_reattach = extract_feature_image(image_to_reattach)
matrix_frame_to_reattach = first_transformation_matrix
for i in range(num_frame_to_reattach):
    matrix_frame_to_reattach = matrix_frame_to_reattach @ list_matrices_common[i]



list_key_frames, matrices_key_frames = compute_key_frames(list_images_common, list_matrices_common, mask)
features_key_frames = compute_features(name_video, list_key_frames, path_images, num_features_image)
#sanity_check_original(frame_to_reattach, features_frame_to_reattach, panorama, features_key_frames, list_key_frames, list_images_common, list_matrices_common)
sanity_check_transformation(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask)

