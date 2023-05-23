import os
import cv2

# list_images_common contains all the images names with correspondent matrices
# path_images is the path to the images folder
# path_matrices is the path to the matrices folder






list_videos = sorted(os.listdir(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset')))   # contains list of videos names
print(list_videos)
for name in list_videos:
    if ('anon' not in name):
        list_videos.remove(name)

for name_video in list_videos:
    print(name_video)
    path_frames = os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images')
    new_path_frames = os.path.join(os.getcwd(), '../final_dataset', name_video, 'images')

    if not os.path.exists(new_path_frames):
        os.makedirs(new_path_frames)

    #mask = mask/255. #binary mask
    mask = cv2.imread(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'mask.png'), 0)
    cv2.imwrite(os.path.join(os.getcwd(), '../final_dataset', name_video, 'mask.png'), mask)

    # path_matrices = os.path.join(os.getcwd(), 'dataset_MICCAI_2020/dataset', name_video,'output_sp_RANSAC')
    # list_images = sorted(os.listdir(path_frames))  # list containing all the images names
    # list_matrices = sorted(os.listdir(path_matrices))  # list containing all the matrices files names
    #
    # list_images_no_extension = []  # list containing all the images names without the extension .png
    # list_matrices_no_extension = []  # list containing all the matrices files names without the extension .txt
    #
    # for name_image in list_images:
    #     name_image = name_image.replace('.png', '')
    #     list_images_no_extension.append(name_image)
    #
    # for name_matrix in list_matrices:
    #     name_matrix = name_matrix.replace('.txt', '')
    #     list_matrices_no_extension.append(name_matrix)
    #
    # list_images_no_extension.sort()
    # list_matrices_no_extension.sort()
    #
    # list_images_common = []  # list containing all the names of images with corresponding matrices
    # for name in list_images_no_extension:
    #     if name in list_matrices_no_extension:
    #         list_images_common.append(name)
    #
    # list_matrices_names_common = []  # list containing all the names of matrices with corresponding images
    # for name in list_matrices_no_extension:
    #     if name in list_images_no_extension:
    #         list_matrices_names_common.append(name)
    #
    # list_images_common.sort()
    # list_matrices_names_common.sort()
    #
    # for elem in list_images_common:
    #     image = cv2.imread(os.path.join(os.getcwd(), path_frames,elem+'.png'), 1)
    #     cv2.imwrite(os.path.join(new_path_frames,elem+'.png'),image)







