import cv2
import numpy as np
from numpy import loadtxt
import os
from skimage.metrics import structural_similarity as ssim
DATASET_FILES_PATH = os.path.join(os.getcwd(), '../../final_dataset_files')
name_experiment = 'Bano_mosaicking'
CANVAS_SHAPE = 4000
NEW_SHAPE = 448



# It is the algorithm for the reconstruction of the panorama based on the given matrices
def panorama_reconstruction(list_images, list_matrices, path_images, name_video, mask, name_experiment):
    path_panorama_image_folder = os.path.join(DATASET_FILES_PATH,'output_panorama_images', name_experiment)
    if not os.path.exists(path_panorama_image_folder):
        # print('creating new panorama image folder')
        os.makedirs(path_panorama_image_folder)

    path_panorama_video_folder = os.path.join(DATASET_FILES_PATH,'output_panorama_video',name_experiment)
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
        path_video = os.path.join(DATASET_FILES_PATH,'output_panorama_video', name_experiment,name_video + '.mp4')
        out = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)

        for name in list_images:
            if ('.DS' in name):
                list_images.remove(name)

        print('panorama creation')
        for index, image_name in enumerate(list_images):
            # print(os.path.join(os.getcwd(),path_images,image_name+'.png'))
            image = cv2.imread(os.path.join(os.getcwd(), path_images, image_name+'.png'), 1)
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


def compute_metric(img1_name, img2_name, mat, path_frames):

    img1 = cv2.imread(os.path.join(path_frames, img1_name+'.png'), 0)
    img2 = cv2.imread(os.path.join(path_frames, img2_name+'.png'), 0)

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





#MAIN FLOW EXECUTION

list_videos = sorted(os.listdir(os.path.join(os.getcwd(), '../../final_dataset')))   # contains list of videos names

for name in list_videos:
    if ('.DS' in name):
        list_videos.remove(name)

for name_video in list_videos:
    print(name_video)
    path_images = os.path.join(os.getcwd(), '../../final_dataset', name_video, 'images')
    path_matrices = os.path.join(os.getcwd(), '../../final_dataset', name_video, 'output_sp_RANSAC')
    mask = cv2.imread(os.path.join(os.getcwd(), '../../final_dataset', name_video, 'mask.png'), 0)
    #mask = mask/255. #binary mask
    mask = cv2.resize(mask, (NEW_SHAPE, NEW_SHAPE))
    shape_image = mask.shape[0]

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


    panorama = panorama_reconstruction(list_images_common, list_matrices_common, path_images, name_video, mask, name_experiment)
    panorama_metric_computation(list_images_common, list_matrices_common, path_images, name_video)
