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

# Class of the used network, derived from a VGG16
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        #eself.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)   # 64x512x2x2
        #out = self.avgpool(out)   # 64x512x7x7
        out = self.flatten(out)   # 64x2048
        #out = self.fc(out)   # 64x2048
        return out



# It extracts features from a given batch using weights of VGG16
def extract_feature_batch(batch_array):
    batch_array = batch_array.to(device)
    with torch.no_grad():

    # Extract the feature from the image
        feature = new_model(batch_array.float())
        feature = feature.cpu().detach().numpy()   #.reshape(-1)
    return feature


# This function extracts the features from a single image and returns a numpy array with these features
def extract_feature_image(img):
    start = datetime.now()
    features = []     # Will contain the feature

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

    # scaler = MinMaxScaler(feature_range=(-1,1))
    # scaler.fit_transform(features)

    features = features.flatten()

    # power normalization, also called square-rooting normalization
    #features = np.sign(features) * np.sqrt(np.abs(features))
    # L2 normalization
    #features = features / np.sqrt(np.dot(features, features))

    v_min, v_max = features.min(), features.max()
    new_min, new_max = -1., 1.
    features = (features - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
    print(datetime.now() - start)
    return features


# It computes feature extraction for all images in the list and creates an external npy file with the features array
def compute_features(name_video, list_images, path_images, num_features_image):
    features_array = np.zeros((len(list_images), num_features_image), dtype=float)
    path_dictionary_folder = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/dictionary_file_npy/Batch+Patch_no_dim_reduction')
    if not os.path.exists(path_dictionary_folder):
        os.makedirs(path_dictionary_folder)

    path_dictionary_npy = os.path.join(path_dictionary_folder, 'dictionary_' + name_video + '.npy')
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
    list_indexes_key_frames = []   # contains the list of the indexes of the key frames
    matrices_key_frames = []   # contains the matrices of the key frames referred to the first image in the centre of the canvas

    path_overlap = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/overlap/'+'overlap_'+name_video)
    if not os.path.exists(path_overlap):
        os.makedirs(path_overlap)

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

    # y = np.array(list_y_points)
    # fig = plt.figure()
    # plt.xlim(0, len(list_images))
    # plt.ylim(0, 100.0)
    # graph, = plt.plot([], [])

    print('LIST INDEXES KEY FRAMES: ', list_indexes_key_frames)

    #ani = FuncAnimation(fig, animate, frames=len(list_images), fargs=[x, y, graph], interval=50, repeat=False)
    # plt.show()
    # path_animation = os.path.join(os.getcwd(), 'dataset_MICCAI_2020_files/animations/animation_overlap/graph_overlap_' + name_video + '.mp4')
    # ani.save(path_animation)

    return list_key_frames, matrices_key_frames



# It adds a transformation to an image to perform sanity checks and returns the transformed image
# with a string indicating the applied transformation
# Possible transformation: rotation, illumination change (contrast and brightness change)
def add_transformation(image, ill_alpha, ill_beta, rot, crop_red):

    # line to apply rotation
    image = imutils.rotate(image, rot)

    # line to apply illumination changes
    image = cv2.convertScaleAbs(image, alpha=ill_alpha, beta=ill_beta)

    #line to apply scaling
    shape = image.shape[0]
    image = image[crop_red:image.shape[1]-crop_red, crop_red:image.shape[0]-crop_red]
    image = cv2.resize(image, (shape, shape))

    return image


# It performs a sanity check between the original images and the frame to reattach with an applied transformation
def sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha, ill_beta, rot, crop_red):
    panorama = panorama * 0.5

    #list frames contain the names of the original frames (with no transformation)
    #features_array_buffer contains the features of the images with transformation applied
    image_to_check = cv2.imread(path_images + '/' + frame_to_reattach + '.png', 1)


    # apply to image_to_check a transformation: rotation, illumination change ir scaling
    image_to_check_transf = add_transformation(image_to_check, ill_alpha, ill_beta, rot, crop_red)
    features_image_to_check = extract_feature_image(image_to_check_transf)

    list_differences = []
    for i in range(features_key_frames.shape[0]):
        dist = np.linalg.norm(features_image_to_check - features_key_frames[i, :])
        list_differences.append(dist)

    #print(list_differences)
    print(list_key_frames)

    min_dist = min(list_differences)
    min_index = list_differences.index(min_dist)
    key_frame_min_dist = list_key_frames[min_index]

    print('ROTATION OF', rot, 'degrees')
    print('CHANGE ILLUMINATION: Contrast is', ill_alpha, 'brightness is', ill_beta)
    print('RESCALING USING A CROP REDUCTION OF', crop_red)
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
    path_folder = os.path.join(os.getcwd(), '../dataset_MICCAI_2020_files/sanity_check_panorama', name_video)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    cv2.imwrite(os.path.join(path_folder, 'sc_rot'+str(rot)+'_ill'+str(ill_alpha)+'_crop'+str(crop_red)+'.png'), panorama_with_key_frame)










#MAIN FLOW EXECUTION

#notwork parameters initialization
#model = models.vgg16_bn(pretrained=True)
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

num_features_image = 346112 #anon001
#num_features_image = 460800 #anon002
#num_features_image = 524288 #anon003
#num_features_image = 739328 #anon005
#num_features_image = 903168 #anon010 and anon012

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



sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=0)
print('---------------------------------------------------------------------------------------------')
sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=10, crop_red=0)
print('---------------------------------------------------------------------------------------------')
sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=5, crop_red=0)
print('---------------------------------------------------------------------------------------------')
sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.10, ill_beta=1.10, rot=0, crop_red=0)
print('---------------------------------------------------------------------------------------------')
sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=0.90, ill_beta=0.90, rot=0, crop_red=0)
print('---------------------------------------------------------------------------------------------')
sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=20)
print('---------------------------------------------------------------------------------------------')
sanity_check(frame_to_reattach, matrix_frame_to_reattach, panorama, features_key_frames, list_key_frames, matrices_key_frames, path_images, mask, ill_alpha=1.0, ill_beta=1.0, rot=0, crop_red=30)
