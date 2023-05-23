import os
import cv2

name_video = 'anon016'
path_video = os.path.join(os.getcwd(), '../final_dataset_files/panorama_mertens/15 fps/anon016.mp4')
cap = cv2.VideoCapture(path_video)

path_images = os.path.join(os.getcwd(), '../final_dataset/anon016/video')
if not os.path.exists(path_images):
    os.makedirs(path_images)

i = 0
while (cap.isOpened()):
  ret, frame = cap.read()
  if ret == False:
    break
  cv2.imwrite(path_images+'/'+name_video+'_'+'{:03d}'.format(i)+'.png', frame)
  i += 1

cap.release()
cv2.destroyAllWindows()




