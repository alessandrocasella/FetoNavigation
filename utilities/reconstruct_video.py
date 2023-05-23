#this script is useful to recreate video from dataset where only frames are available

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def reconstruct_video(frames_list, path_frames, shape, path_final_video):

    #initialization of video parameters
    fps = 25.0
    video= cv2.VideoWriter(path_final_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (shape, shape))

    #video creation
    for count in range(len(frames_list)):
        print(count)
        image = cv2.imread(path_frames+ '/' + frames_list[count], 1)
        try:
            video.write(image)
        except:
            print(frames_list[count])
        #plt.imshow(image)
        #plt.show()

    video.release()
    cv2.destroyAllWindows()



#MAIN FLOW EXECUTION
# name_video_list = sorted(os.listdir(os.path.join(os.getcwd(), 'video')))   #contains the list of the names of the videos
# #print(name_video_list)
# path_video_list = []   #contains the list of the paths of all videos
#
# for i in name_video_list:
#     if ('frame' not in i):
#         name_video_list.remove(i)

# for name_video in name_video_list:
#     #print(i)
#     if os.path.isdir(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video)):
#         path_video_list.append(os.path.join(os.getcwd(), '../dataset_MICCAI_2020/dataset', name_video, 'images'))
#print(path_video_list)

#for count in range(len(name_video_list)):

path_final_video = os.path.join(os.getcwd(), 'anon002.mp4')
if not os.path.exists(path_final_video):
    path_frames = os.path.join(os.getcwd(), '../final_dataset/anon002/images')
    # name_video = name_video_list[count]
    # print(name_video)
    #print(path_frames)
    frames_list = sorted(os.listdir(path_frames))   # list of frames
    for i in frames_list:
        if ('frame' not in i):
            print(i, 'REMOVED')
            frames_list.remove(i)

    img = cv2.imread(path_frames+ '/' + frames_list[0], 1)
    shape = img.shape[0]
    reconstruct_video(frames_list, path_frames, shape, path_final_video)