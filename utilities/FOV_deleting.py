# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

# Code adapted from a version developed by https://github.com/ChiaraLena

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import math
import os
import cv2.cv2 as cv2
import numpy as np
from PIL import Image
from matplotlib.widgets import Button
from matplotlib.widgets import EllipseSelector




points_ellipse = []
global_list = []
global_square_points = [(0,0),(0,0)]
global_ellipse_coord = (((0,0),(0,0)), (0,0))
global_flag = False
global_name_videoclip = ''

# FUNCTIONS TO DETECT automatically THE FOV
def image_hough(im_gray):
    im_gray = cv2.medianBlur(im_gray, 5)
    rows = im_gray.shape[0]
    circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT, 1, rows,
                               param1=80, param2=20,
                               minRadius=150, maxRadius=0)
    #cv2.imwrite(os.path.join(os.getcwd(), 'ellipse.png'), circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(im_gray, (i[0], i[1]), i[2], (255, 0, 0), 4)
            # draw the center of the circle
            cv2.circle(im_gray, (i[0], i[1]), 2, (255, 0, 0), -1)

        center_x = circles[0, 0, 0]
        center_y = circles[0, 0, 1]
        radius = circles[0, 0, 2]
    else:
        center_x = 0
        center_y = 0
        radius = 0

    return im_gray, center_x, center_y, radius

def image_analyze_fov(list):
    img = list[0]
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_gray_to_be_processed = image_gray.copy()
    image_gray_hough, center_x, center_y, radius = image_hough(image_gray_to_be_processed)
    return image_gray_hough, center_x, center_y, radius


# FUNCTIONS TO DETECT THE EVENTS IN THE INTERACTIVE PLOT
def onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
    points_ellipse.append((eclick.xdata, eclick.ydata))
    print('sono in onselect: ', points_ellipse)
    print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
    points_ellipse.append((erelease.xdata, erelease.ydata))
    print('used button  : ', eclick.button)

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.ES.active:
        print('EllipseSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.ES.active:
        print('EllipseSelector activated.')
        toggle_selector.ES.set_active(True)

def on_close(event):
    print('onclose')
    global points_ellipse
    global global_list
    global global_square_points
    global global_ellipse_coord
    global global_flag
    preprocessed_images = []
    if len(points_ellipse) == 0:
        print('ERROR: la lista di punti è vuota')
    else:
        img = global_list[0]
        im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # only image
        height, width = im_gray.shape

        print(points_ellipse)
        axis_x = int((points_ellipse[-1][0] - points_ellipse[-2][0]) // 2)
        axis_y = int((points_ellipse[-1][1] - points_ellipse[-2][1]) // 2)
        axes_length = (axis_x, axis_y)

        # the coordinates obtained are referred to a squared image --> translation to the original one
        new_x_C = points_ellipse[-1][0] + (width - height) // 2
        new_x_A = points_ellipse[-2][0] + (width - height) // 2

        # now calculate the parameters of the ellipse in the general case of the original image
        center_x = int((new_x_A + new_x_C) // 2)
        center_y = int((points_ellipse[-1][1] + points_ellipse[-2][1]) // 2)
        center_ellipse = (center_x, center_y)
        # print(center_ellipse)
        ellipse_coord = (center_ellipse, axes_length)
        print('now i print the type of ellipse coord')
        print(type(ellipse_coord))
        print(ellipse_coord)


        # circumscribed square with radius = max axis
        print('lunghezza assi: ', axes_length)
        max_axis = max(axis_x, axis_y)
        print('asse max: ', max_axis)

        x_left = int(center_x - max_axis)
        y_left = int(center_y - max_axis)

        x_right = int(center_x + max_axis)
        y_right = int(center_y + max_axis)


        square_points = [(x_left, y_left), (x_right, y_right)]
        global_square_points = square_points
        global_ellipse_coord = ellipse_coord
        print('sono in onclose e le coordinate sono: ', global_ellipse_coord)
        global_flag = True
        list_images = preprocessing_image_ellipse(global_list)
        for i in list_images:
            preprocessed_images.append(i)
        global_list = preprocessed_images
        print(len(preprocessed_images))
        return


# SERIES OF FUNCTIONS TO PREPROCESS IMAGES WITH ELLIPTICAL FOV:
# square_ellipse_selection show the automatically elliptical fov detected:
#       Automatic detection was successful -> image_preprocessing
#           - image_preprocessing applies to all images the function circumscribed_square
#           - circumscribed_square cuts the images
#       Automatic detection wasn't successful ->  square_in_ellipse_detection
#           - square_in_ellipse_detection returns the ellipse coordinates drawn by the user
#           - preprocessing_image_ellipse applies to all images the function circumscribed_square_ellipse
#           - circumscribed_square_ellipse cuts the images

def square_in_ellipse_detection(list):
    # print('square detection')
    global global_list
    img = list[0]
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = im_gray.shape
    x_left = (width - height) // 2
    x_right = width - x_left

    im_gray_squared = im_gray[0:height, x_left:x_right]
    height_new, width_new = im_gray_squared.shape

    x = np.arange(width_new)
    y = np.arange(height_new)
    fig, ax = plt.subplots()
    ax.imshow(im_gray_squared, cmap='gray')
    plt.show()

    global_list = list
    toggle_selector.ES = EllipseSelector(ax, onselect, interactive=True, drawtype='box')
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    #plt.waitforbuttonpress()
    fig.canvas.mpl_connect('close_event', on_close)

def circumscribed_square_ellipse(image, square_points):
    circumscribed_square = image[square_points[0][1]:square_points[1][1], square_points[0][0]:square_points[1][0]]
    return circumscribed_square

def preprocessing_image_ellipse(list):
    # print('Entering in preprocessing ellipse')
    global global_ellipse_coord
    global global_square_points

    mask = np.zeros(list[0].shape, dtype=np.uint8)
    cv2.ellipse(mask, global_ellipse_coord[0], global_ellipse_coord[1], 0, 0, 360, (255, 255, 255), -1)
    plt.imshow(mask)
    plt.show()

    path_masks = os.path.join(os.getcwd(), 'FOV_deleted', global_name_videoclip+'.png')
    cv2.imwrite(path_masks, mask)

    preprocessed_images = []
    for elem in list:
        try:
            squared_image = circumscribed_square_ellipse(elem, global_square_points)
            # plt.imshow(squared_image, cmap='gray')
            # plt.show()
            preprocessed_images.append(squared_image)
        except:
            print("ERROR")
    return preprocessed_images

def circumscribed_square(image, center_x, center_y, radius):

    x_left = int(center_x - radius)
    y_left = int(center_y - radius)
    x_right = int(center_x + radius)
    y_right = int(center_y + radius)

    circumscribed_square = image[y_left:y_right, x_left:x_right]  # put the image in the center
    # plt.imshow(circumscribed_square, cmap='gray')
    # plt.show()
    #height, width = circumscribed_square.shape
    return circumscribed_square

def image_preprocessing(center_x, center_y, radius, list):
    print('Entering in image_preprocessing')
    global global_list
    global global_flag


    mask = np.zeros(list[0].shape, dtype=np.uint8)
    mask = cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    # plt.imshow(mask)
    # plt.show()

    path_masks = os.path.join(os.getcwd(), 'FOV_deleted', global_name_videoclip+'.png')
    cv2.imwrite(path_masks, mask)

    preprocessed_images = []
    for elem in list:
        squared_image = circumscribed_square(elem, center_x, center_y, radius)   # image in the center
        # plt.imshow(squared_image, cmap='gray')
        # plt.show()
        if squared_image is not None:
            preprocessed_images.append(squared_image)
        else:
            print('Circle not found')
    global_list = preprocessed_images
    global_flag = True

def square_ellipse_selection(img, center_x, center_y, radius, list):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.3)
    ax.imshow(img, cmap='gray')

    axButton_YES = plt.axes([0.2, 0.1, 0.2, 0.1])
    button_yes = Button(ax=axButton_YES, label="YES", color='teal', hovercolor='tomato')

    axButton_NO = plt.axes([0.6, 0.1, 0.2, 0.1])
    button_no = Button(ax=axButton_NO, label="NO", color='teal', hovercolor='tomato')


    def click_YES(event):
        # print("you have clicked YES")
        plt.close()
        image_preprocessing(center_x, center_y, radius, list)


    def click_NO(event):
        print("you have clicked NO")
        global global_ellipse_coord
        plt.close()
        square_in_ellipse_detection(list)

    button_yes.on_clicked(click_YES)
    button_no.on_clicked(click_NO)
    plt.show()
    return


# FUNCTION THAT STARTS THE PRE-PROCESSING--> this must be done BEFORE entering into the network
def pre_processing(list):
    print('Pre Processing: FOV detection and elimination')
    image_grey_hough, center_x, center_y, radius = image_analyze_fov(list)
    square_ellipse_selection(image_grey_hough, center_x, center_y, radius, list)
    while not global_flag:
        pass
    global global_list
    return global_list



#CODICE MIO PER PRENDERE I FRAME IN INGRESSO
name_patient_list = os.listdir('processed_frames')
name_patient_list.sort()

name_video_list = []
path_video_list = []

for i in name_patient_list:
    if ('Video' not in i):
        name_patient_list.remove(i)

for i in name_patient_list:
    #print(i)
    folder_list = sorted(os.listdir('processed_frames/' + i))

    for elem in folder_list:
            if os.path.isdir(os.path.join(os.getcwd(), 'processed_frames', i, elem)):
                name_video_list.append(elem)
                path_video_list.append('processed_frames/' + i + '/' + elem + '/images')

for path in path_video_list:
    list_frames_video = []
    list_name_frames = sorted(os.listdir(path))
    for frame in list_name_frames:
        #print(path+frame)
        image = cv2.imread(path+'/'+frame)
        list_frames_video.append(image) #Primo parametro da passare a pre-processing
    for name in name_video_list:
        if name in path:
            global_name_videoclip = name #Secondo parametro da passare a pre-processing

    pre_processing(list_frames_video)








#  Pre_processing:
#  1. image_analyze_fov -> image_hough -> return center coordimates, radius and the image with the ellipse
#  2. square_ellipse_selection:
#        a. click_yes -> image_preprocessing -> circumscribe_square
#        b. click_no -> square_in_ellipse_detection -> onclose -> preprocessing_image_ellipse -> circumscribed_square_ellipse
