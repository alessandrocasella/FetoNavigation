import cv2
import os



name_video = 'anon024_CLIP01'

path_folder_old = os.path.join(os.getcwd(), '../final_dataset', name_video, 'images')
path_folder_new = os.path.join(os.getcwd(), '../final_dataset', name_video, 'images_old_size')
if not os.path.exists(path_folder_new):
    os.rename(path_folder_old, path_folder_new)

list_images_names = sorted(os.listdir(path_folder_new))

if not os.path.exists(path_folder_old):
    os.makedirs(path_folder_old)

    for image_name in list_images_names:
        image = cv2.imread(os.path.join(path_folder_new, image_name), 1)
        image = cv2.resize(image, (470, 470), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(path_folder_old, image_name), image)


path_mask_old = os.path.join(os.getcwd(), '../final_dataset', name_video, 'mask.png')
path_mask_new = os.path.join(os.getcwd(), '../final_dataset', name_video, 'mask_old_size.png')
if not os.path.exists(path_mask_new):
    os.rename(path_mask_old, path_mask_new)
    mask = cv2.imread(path_mask_new, 1)
    mask = cv2.resize(mask, (470, 470), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path_mask_old, mask)



