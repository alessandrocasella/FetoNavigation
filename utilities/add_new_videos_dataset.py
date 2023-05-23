import os
import cv2

# path_video = os.path.join(os.getcwd(),'dataset_MICCAI_2021/anon016/Video016_CLIP16/images')
# name_new_folder = 'anon016'
#
# new_path_frames = os.path.join(os.getcwd(), 'final_dataset', name_new_folder, 'images')
#
# if not os.path.exists(new_path_frames):
#     os.makedirs(new_path_frames)
#
# list_frames = sorted(os.listdir(os.path.join(os.getcwd(),path_video)))
#
# for elem in list_frames:
#     image = cv2.imread(os.path.join(path_video,elem),1)
#     cv2.imwrite(os.path.join(new_path_frames,elem),image)


# path_video = os.path.join(os.getcwd(),'dataset_MICCAI_2021/Video018/Video018_CLIP03/images')
# name_new_folder = 'anon018'
#
# new_path_frames = os.path.join(os.getcwd(), 'final_dataset', name_new_folder, 'images')
#
# if not os.path.exists(new_path_frames):
#     os.makedirs(new_path_frames)
#
# list_frames = sorted(os.listdir(os.path.join(os.getcwd(),path_video)))
#
# for elem in list_frames:
#     image = cv2.imread(os.path.join(path_video,elem),1)
#     cv2.imwrite(os.path.join(new_path_frames,elem),image)




# path_video = os.path.join(os.getcwd(),'dataset_MICCAI_2021/Video020/Video020_CLIP01/images')
# name_new_folder = 'anon020'
#
# new_path_frames = os.path.join(os.getcwd(), 'final_dataset', name_new_folder, 'images')
#
# if not os.path.exists(new_path_frames):
#     os.makedirs(new_path_frames)
#
# list_frames = sorted(os.listdir(os.path.join(os.getcwd(),path_video)))
#
# for elem in list_frames:
#     image = cv2.imread(os.path.join(path_video,elem),1)
#     cv2.imwrite(os.path.join(new_path_frames,elem),image)



path_video = os.path.join(os.getcwd(), '../dataset_MICCAI_2021/Video022/Video022_CLIP02/images')
name_new_folder = 'anon022'

new_path_frames = os.path.join(os.getcwd(), '../final_dataset', name_new_folder, 'images')

if not os.path.exists(new_path_frames):
    os.makedirs(new_path_frames)

list_frames = sorted(os.listdir(os.path.join(os.getcwd(),path_video)))

for elem in list_frames:
    image = cv2.imread(os.path.join(path_video,elem),1)
    cv2.imwrite(os.path.join(new_path_frames,elem),image)