import os
import cv2

images_path = os.path.join(os.getcwd(), 'final_dataset/anon025/images')
list_frames = sorted(os.listdir(os.path.join(images_path)))

new_images_path = os.path.join(os.getcwd(), 'final_dataset/anon025/images_new')
if not os.path.exists(new_images_path):
    os.makedirs(new_images_path)


for image_name in list_frames:
    frame = cv2.imread(os.path.join(images_path,image_name),1)
    border = (frame.shape[1] - frame.shape[0]) // 2
    frame = cv2.copyMakeBorder(frame, border, border, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imwrite(os.path.join(new_images_path, image_name),frame)

mask = cv2.imread(os.path.join(os.getcwd(), 'final_dataset/anon025/mask.png'),1)
border = (mask.shape[1] - mask.shape[0]) // 2
mask = cv2.copyMakeBorder(mask, border, border, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
cv2.imwrite(os.path.join(os.getcwd(), 'final_dataset/anon025/mask_new.png'), mask)


