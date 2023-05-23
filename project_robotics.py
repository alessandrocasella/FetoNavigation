
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import torch
#import torch.cuda.profiler as profiler
#import pyprof
#pyprof.init()
from  matplotlib.widgets import Button
import cv2
import numpy as np
import random
import math
from numpy.linalg import inv
import os
from matplotlib.widgets import EllipseSelector
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gc


def image_hough(im_gray):
    im_gray = cv2.medianBlur(im_gray, 5)
    rows = im_gray.shape[0]
    circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT, 1, rows,
                               param1=80, param2=20,
                               minRadius=150, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(im_gray, (i[0], i[1]), i[2], (255, 0, 0), 4)
            # draw the center of the circle
            cv2.circle(im_gray, (i[0], i[1]), 2, (255, 0, 0), -1)
    return im_gray

# SERIES OF FUNCTION TO PREPROCESS IMAGES WITH SQUARE FOV
def crop_center(img, crop_width, crop_height):
    img_width, img_height = img.size
    return img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def rotate(origin, point, angle):
        # Rotate a point counterclockwise by a given angle around a given origin. The angle should be given in radians.
        angle = math.radians(angle)
        ox, oy = origin
        px, py = point
        qx = int(ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy))
        qy = int(oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy))
        return qx, qy

def inscribed_square(im_gray):

    # preprocessing: median filter  (to reduce noise)
    im_gray = cv2.medianBlur(im_gray, 5)

    rows = im_gray.shape[0]
    circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT, 1, rows,
                               param1=80, param2=20,
                               minRadius=150, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(im_gray, (i[0], i[1]), i[2], (0, 255, 0), 4)
            # draw the center of the circle
            cv2.circle(im_gray, (i[0], i[1]), 2, (0, 128, 255), -1)
    else:
        return None

    # inscribed square
    center_x = circles[0, 0, 0]
    center_y = circles[0, 0, 1]
    radius = circles[0, 0, 2]
    side = radius * math.sqrt(2) // 2

    x_left = int(center_x - side)
    y_left = int(center_y - side)
    x_right = int(center_x + side)
    y_right = int(center_y + side)

    inscribed_square = im_gray[y_left:y_right, x_left:x_right]
    height, width = inscribed_square.shape
    return inscribed_square

# image_processing includes all the preprocessing that is before the CDA in case of square FOV
def image_preprocessing(path):
    lst = os.listdir(path)
    preprocessed_images = []
    number_circles_not_found = 0
    lst.sort()

    for i in lst:
        if ('jpg' not in i):
            lst.remove(i)

    for i in lst:
        test_image = cv2.imread(path + i)
        grey_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        squared_image = inscribed_square(grey_image)
        if (squared_image) is not None:
            preprocessed_images.append(squared_image)
        else:
            print ('Circle not found')
            number_circles_not_found += 1
    print('number of circles not found: ')
    print(number_circles_not_found)
    print('number of circles found:')
    print(len(preprocessed_images))
    return preprocessed_images




# SERIES OF FUNCTIONS TO PREPROCESS IMAGES WITH ELLIPTICAL FOV
def square_in_ellipse_detection(path):
    points_ellipse = []
    def onselect(eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        #print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        points_ellipse.append((eclick.xdata, eclick.ydata))
        #print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        points_ellipse.append((erelease.xdata, erelease.ydata))
        #print('used button  : ', eclick.button)

    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.ES.active:
            print('EllipseSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.ES.active:
            print('EllipseSelector activated.')
            toggle_selector.ES.set_active(True)

    img = cv2.imread(path+'frame000.jpg',1)
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = im_gray.shape
    x_left = (width-height) // 2
    x_right = width - x_left

    im_gray_squared = im_gray[0:height, x_left:x_right]
    height_new, width_new = im_gray_squared.shape

    x = np.arange(width_new)
    y = np.arange(height_new)
    fig, ax = plt.subplots()
    ax.imshow(im_gray_squared, cmap='gray')

    toggle_selector.ES = EllipseSelector(ax, onselect, drawtype='box', interactive=True)
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show()

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
    #print(center_ellipse)
    ellipse_coord = (center_ellipse, axes_length)
    print('now i print the type of ellipse coord')
    print(type(ellipse_coord))
    print(ellipse_coord)

    image_with_ellipse = cv2.ellipse(im_gray, center_ellipse, axes_length, 0, 0, 360, 255, 2)
    plt.imshow(image_with_ellipse, cmap='gray')
    plt.show()

    # inscribed square with radius = minor axis
    min_axis = min(axis_x, axis_y)
    side = int(min_axis * math.sqrt(2) // 2)

    x_left = int(center_x - side)
    y_left = int(center_y - side)

    x_right = int(center_x + side)
    y_right = int(center_y + side)

    im_gray_copy = im_gray.copy()
    image_with_square = cv2.rectangle(im_gray_copy, (x_left, y_left), (x_right, y_right), (0, 255, 0), 2)

    inscribed_square = im_gray[y_left:y_right, x_left:x_right]
    square_points = [(x_left, y_left), (x_right, y_right)]
    return square_points, ellipse_coord

def inscribed_square_ellipse(im_gray, square_points):
    inscribed_square = im_gray[square_points[0][1]:square_points[1][1], square_points[0][0]:square_points[1][0]]
    return inscribed_square

def image_preprocessing_ellipse(path):
    lst = os.listdir(path)
    lst.sort()
    preprocessed_images = []
    square_points, ellipse_coord = square_in_ellipse_detection(path)
    print(ellipse_coord)

    for i in lst:
        if ('jpg' not in i):
            lst.remove(i)

    for i in lst:
        try:
            test_image = cv2.imread(path + i)
            grey_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
            squared_image = inscribed_square_ellipse(grey_image, square_points)
            preprocessed_images.append(squared_image)
        except:
            print("ERROR")
    return preprocessed_images, ellipse_coord


#-------------------------------------------------------------------------------------------------------
def image_analyze_fov (path):
    img = plt.imread(path+'frame000.jpg')
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_gray_to_be_processed = image_gray.copy()
    image_gray_hough = image_hough(image_gray_to_be_processed)
    return image_gray_hough, path

def square_ellipse_selection(img, path) :
    fig, ax= plt.subplots()
    plt.subplots_adjust(left = 0.1, bottom = 0.3)
    ax.imshow(img, cmap='gray')

    axButton_YES = plt.axes([0.2, 0.1, 0.2, 0.1])
    button_yes = Button(ax=axButton_YES, label= "YES", color = 'teal', hovercolor= 'tomato')

    axButton_NO = plt.axes([0.6, 0.1, 0.2, 0.1])
    button_no = Button(ax=axButton_NO, label= "NO", color = 'teal', hovercolor= 'tomato')
    preprocessed_images = []
    list_ellipse_coord = []

    def click_YES(event):
        #print("you have clicked YES")
        plt.close()
        list_images = image_preprocessing(path)
        for i in list_images :
            preprocessed_images.append(i)

    def click_NO(event):
        #print("you have clicked NO")
        plt.close()
        list_images, ellipse_coord = image_preprocessing_ellipse(path)
        list_ellipse_coord.append(ellipse_coord)
        for i in list_images :
            preprocessed_images.append(i)

    button_yes.on_clicked(click_YES)
    button_no.on_clicked(click_NO)
    plt.show()
    ellipse_coord = list_ellipse_coord[0]
    return preprocessed_images, ellipse_coord
#-------------------------------------------------------------------------------------------------------

def CDA(grey_image, list_datum):
    image_height, image_width = grey_image.shape
    patch_size = 64 #128

    for i in range(100):
        rho_x = random.randint(22, image_width-22-patch_size)
        rho_y = random.randint(22, image_width-22-patch_size)

        top_point = (rho_x, rho_y)
        left_point = (top_point[0], patch_size + top_point[1])
        bottom_point = (patch_size + top_point[0], patch_size + top_point[1])
        right_point = (patch_size + top_point[0], top_point[1])
        four_points = [top_point, left_point, bottom_point, right_point]


        #TRANSLATION
        translx = random.randint(-16, 16);
        #print ('translation in x value for patch '+ str(i) +' =' +  str(translx))

        transly = random.randint(-16, 16);
        #print ('translation in y value for patch '+ str(i) +' =' +  str(transly))

        top_point_trasl = (top_point[0] + translx, top_point[1] + transly)
        right_point_trasl = (patch_size + top_point_trasl[0], top_point_trasl[1])
        bottom_point_trasl = (patch_size + top_point_trasl[0], patch_size + top_point_trasl[1])
        left_point_trasl = (top_point_trasl[0], patch_size + top_point_trasl[1])
        perturbed_four_points = [top_point_trasl, left_point_trasl, bottom_point_trasl, right_point_trasl]


        #ROTATION
        x_center = (right_point_trasl[0] - top_point_trasl[0]) // 2
        y_center = (left_point_trasl[1] - top_point_trasl[1]) // 2
        origin = (x_center + top_point_trasl[0], y_center + top_point_trasl[1])

        angle = random.uniform(-5,5)

        top_point_new = rotate(origin, top_point_trasl, angle)
        bottom_point_new = rotate(origin, bottom_point_trasl, angle)
        left_point_new = rotate(origin, left_point_trasl, angle)
        right_point_new = rotate(origin, right_point_trasl, angle)
        final_four_points = [top_point_new, left_point_new, bottom_point_new, right_point_new]


        H, _ = cv2.estimateAffine2D(np.float32(four_points), np.float32(final_four_points))
        H_inverse = cv2.invertAffineTransform(H)

        height, width = grey_image.shape
        warped_image = cv2.warpAffine(grey_image, H_inverse, (height, width))


        Ip1 = grey_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
        Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

        #cv2.imwrite('Resources/Patches/patch1_1.' + str(i) +'.png', Ip1)
        #cv2.imwrite('Resources/Patches/patch1_2.'+ str(i) +'.png', Ip2)

        training_image = np.dstack((Ip1, Ip2))
        H_four_points = np.subtract(np.array(final_four_points), np.array(four_points))

        datum= (training_image, H_four_points)
        list_datum.append(datum)

    return list_datum, image_width, patch_size

def save_external_file(list):
    out = np.empty(len(list), dtype=object)
    out[:] = list
    return out


#MAIN FLOW EXECUTION FOR PRE-PROCESSING OF THE TRAINING SET--> this must be done BEFORE entering into the network
def pre_processing(path, path_processed):
    list_name_folders = os.listdir(path)
    list_processed = os.listdir(path_processed)
    list_name_folders.sort()
    list_processed.sort()

    for i in list_name_folders:
        if ('frames' not in i):
            list_name_folders.remove(i)
        list_name_folders = list(map(lambda x: x.replace('frames from ', ''), list_name_folders))

    if (len(list_processed) != 0) :
        for i in list_processed:
            if ('file' not in i):
                list_processed.remove(i)

    list_folder_coord = []
    if (len(list_processed) != len(list_name_folders)) :
        for name in list_name_folders:
            image_grey_hough, path_returned = image_analyze_fov(path + 'frames from ' + name + '/')

            preprocessed_images, ellipse_coord = square_ellipse_selection(image_grey_hough, path_returned)
            print('sono in preprocessing: ')
            print(ellipse_coord)
            datum = (name, ellipse_coord)
            list_folder_coord.append(datum)

            out = save_external_file(preprocessed_images)

            new_path = path_processed + "file_" + name
            np.save(new_path, out)
    else:
        print('all files already present')

    file_folder_coord = save_external_file(list_folder_coord)
    print((file_folder_coord.shape))
    np.save('folder_coord', file_folder_coord)


#flag to avoid the pre-processing as i know the file folder is complete
LOAD_IMAGES = False
if (LOAD_IMAGES == True):
    pre_processing(os.path.join(os.getcwd(),'frames from videos/'), os.path.join(os.getcwd(),'processed/'))

#FINISH OF PREPROCESSING PART FOR TRAINING SET
#
#
#
#
#
#
#
#
#
#
# NOW THERE IS THE NETWORK PART WITH CDA
def save_data(path):
    list_datum = []
    print(path)
    file = np.load(path, allow_pickle=True)
    for elem in file:
       list_datum, image_width, patch_size = CDA(elem, list_datum)
    random.shuffle(list_datum)
    out = np.empty(len(list_datum), dtype=object)
    out[:] = list_datum
    del file
    gc.collect()
    return out, image_width, patch_size

#class to define the network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1),
                                    nn.BatchNorm2d(64, affine=True),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64, affine=True),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64, affine=True),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64, affine=True),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128, affine=True),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128, affine=True),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128, affine=True),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128, affine=True),
                                    nn.ReLU())
        self.fc1 = nn.Linear(128 * 8 * 8, 1024) # 128 * 16 * 16
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.reshape(-1, 128 * 8 * 8) # 128 * 16 * 16
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#class to prepare the dataset (splitting x and y) --> works ONLY with np array
class PrepareDataset(Dataset):
    def __init__(self, array_images, image_width, patch_size):
        X = ()
        Y = ()
        # lst = os.listdir(path)
        it = 0
        norm_factor = 32
        array = array_images

        for i in range(len(array)):
            x = (array[i][0].astype(float) - 127.5) / 127.5
            x = np.swapaxes(x, 0, 2)
            x = torch.from_numpy(x)
            X = X + (x,)
            y = torch.from_numpy(array[i][1].astype(float) / norm_factor)
            Y = Y + (y,)
            it += 1
        self.len = it
        self.X_data = X
        self.Y_data = Y

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return self.len



#IMPLEMENTATION OF THE NETWORK
batch_size = 100
criterion = nn.MSELoss()
epochs = 10000
model = Model().to(device, non_blocking=True)
summary(model,(2,64,64))
optimizer = optim.SGD(model.parameters(),lr=0.005, momentum=0.9)
j = 0

LOAD = True # if false it executes the network. if true it uploads an existing model

#execution of the network (only if the flag LOAD = False, else skipped)
if (LOAD == False):
    path = os.path.join(os.getcwd(), 'processed/')
    list_files = os.listdir(path)

    for file in list_files:
        if ('file' not in file):
            list_files.remove(file)

    for epoch in trange(epochs, position=0, leave=True):
        random.shuffle(list_files)
        for file in tqdm(list_files, position=0, leave=True):
            print(file)
            output_array, image_width, patch_size = save_data(path + file)
            training_data = PrepareDataset(output_array, image_width, patch_size)
            TrainLoader = DataLoader(training_data, batch_size, pin_memory=True, num_workers=4)
            num_samples = training_data.len
            steps_per_epoch = num_samples / batch_size

            for i, (images, target) in enumerate(tqdm(TrainLoader, position=0, leave=True)):
                optimizer.zero_grad()
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                # images = images.permute(0,3,1,2).float();
                images = images.float()
                target = target.float()
                outputs = model(images)
                loss = criterion(outputs, target.view(-1, 8))
                loss.backward()
                optimizer.step()

            print('i have completed file ' + file)
            del output_array
            del images
            del training_data
            del TrainLoader
            gc.collect()
        # print('I have completed the epoch')
        print('Train Epoch: [{}/{}]: Mean Squared Error: {:.6f}'.format(
            epoch + 1, epochs, loss))
        if (epoch % 50) == 0:
            intermediate_state = {'epoch': epochs, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
            torch.save(intermediate_state,
                       os.path.join(os.getcwd(), 'intermediate_models/intermediate_model' + str(epoch) + '.pth'))

    state = {'epoch': epochs, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, 'DeepHomographyEstimation.pth')

else:
    PATH = os.path.join(os.getcwd(),'intermediate_model350.pth')
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

# RECONSTRUCTION PART --------------------------------------------------------------------------------------------

#PREPROCESSING FOR TEST SET

def patch_extraction_test(gray_image_1, gray_image_2, list_datum, min_dimension_image):

    image_width = min_dimension_image
    patch_size = 64 # 128

    for i in range(100):
        rho_x = random.randint(0, image_width-patch_size)
        rho_y = random.randint(0, image_width-patch_size)

        top_point = (rho_x, rho_y)
        left_point = (top_point[0], patch_size + top_point[1])
        bottom_point = (patch_size + top_point[0], patch_size + top_point[1])
        right_point = (patch_size + top_point[0], top_point[1])

        x = [top_point[0], right_point[0], left_point[0], bottom_point[0]]
        y = [top_point[1], right_point[1], left_point[1], bottom_point[1]]

        Ip1 = gray_image_1[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
        Ip2 = gray_image_2[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

        testing_image = np.dstack((Ip1, Ip2))

        datum = (testing_image,x,y)
        list_datum.append(datum)

    return list_datum

def DLT(H4p, x, y):

    xd = np.array([0., 0., 0., 0.])
    yd = np.array([0., 0., 0., 0.])

    for i in range(len(H4p)):
        xd[i] = H4p[i][0] + x[i]
        yd[i] = H4p[i][1] + y[i]

    npoints = 4
    A = []
    for i in range(npoints):
        A.append([x[i], y[i], 1, 0, 0, 0, -xd[i] * x[i], -xd[i] * y[i], -xd[i]])
        A.append([0, 0, 0, x[i], y[i], 1, -yd[i] * x[i], -yd[i] * y[i], -yd[i]])

    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, : ] / Vh[-1, -1]
    #print(L)
    H = np.reshape(L, (3,3))
    #print(H)
    return H

def prediction_matrix(path):

    data_test = np.load(path, allow_pickle=True)

    min_dimension_image = 2000
    for elem in data_test:
        if (elem.shape[0] < min_dimension_image):
            min_dimension_image = elem.shape[0]

    list_H_matrices = []
    for index in range(len(data_test)-1):
        list_datum = []
        list_datum = patch_extraction_test(data_test[index], data_test[index+1], list_datum, min_dimension_image)
        out = np.empty(len(list_datum), dtype=object )
        out[:] = list_datum

        #print(out[0][0].shape) #STACK DELLE 2 IMMAGINI -->128x128x2
        #print(len(out[0][1])) #LISTA COORDINATE X --> 4
        #print(len(out[0][2])) #LISTA COORDINATE Y --> 4

        #creation of a numpy array out_copy that has the same content as the fist element of the tuple --> it contains just the stack of the images
        out_copy = np.empty(len(out), dtype=object )
        for i in range(len(out_copy)):
            out_copy[i] = out[i][0]

        #normalization and swapaxes for out_copy
        for i in range(len(out_copy)):
            out_copy[i] = (out_copy[i].astype(float) - 127.5) / 127.5
            out_copy[i] = np.swapaxes(out_copy[i], 0, 2)


        #creation of an array of tuples where each tuple contains the stack normalized and swapped, x and y
        output_data = np.empty(len(out), dtype=object )
        list_output = []
        for i in range(len(out)):
            datum_final = (out_copy[i], out[i][1], out[i][2])
            list_output.append(datum_final)
        output_data[:] = list_output

        del(out)
        del(out_copy)


        #prediction and reconstruction part
        #list_matrices = []
        list_tx = []
        list_ty = []
        list_R = []
        with torch.autograd.profiler.emit_nvtx():
            for i in tqdm(range(len(output_data))):

                # make example a torch tensor
                value = torch.from_numpy(output_data[i][0])

                # then put it on the GPU, make it float and insert a fake batch dimension
                test_value = Variable(value.cuda())
                test_value = test_value.float()
                test_value = test_value.unsqueeze(0)

                # pass it through the model
                prediction = model(test_value)

                # get the result out and reshape it
                cpu_pred = prediction.cpu()
                result = cpu_pred.data.numpy()
                #list_matrices.append(result)
                norm_factor = 32
                H4p_result = np.reshape(result, (4, 2))*norm_factor

                H4p_orig = np.array([[0, 0],
                                     [128, 0],
                                     [0, 128],
                                     [128, 128]])


                H4p = np.add(H4p_orig,H4p_result)

                H_warp, _ = cv2.estimateAffine2D(np.float32(H4p_orig), np.float32(H4p))
                #H_warp = cv2.invertAffineTransform(H_warp)
                #K = np.array([1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0])
                tx, ty = H_warp[0, 2], H_warp[1, 2]
                #w, u, vt = cv2.SVDecomp(H_warp)
                #R = vt @ u
                R = H_warp[0:2, 0:2]
                rotation = np.arctan2(R[1, 0], R[0, 0])*(180/np.pi)

                list_tx.append(tx), list_ty.append(ty), list_R.append(rotation)


        tx = np.median(list_tx, axis=0)
        ty = np.median(list_ty, axis=0)
        R = np.median(list_R, axis=0)*np.pi/180.0
        list_H_matrices.append([tx, ty, R])
        #array_matrices = np.concatenate(list_matrices, axis=0)
        #result_median = np.median(array_matrices, axis=0)
        #result_median = result_median * norm_factor
        #result_H4p = np.reshape(result_median, (4, 2))

        #code to swap third and fourth rows
        #temp_x = result_H4p[2][0]
        #temp_y = result_H4p[2][1]
        #result_H4p[2][0] = result_H4p[3][0]
        #result_H4p[2][1] = result_H4p[3][1]
        #result_H4p[3][0] = temp_x
        #result_H4p[3][1] = temp_y

        #H_matrix = DLT(result_H4p, output_data[i][1], output_data[i][2]) #H_matrix is the 3x3 matrix i need to perform blending
        #list_H_matrices.append(result_H4p)

    return list_H_matrices #this list is made of 4x2 matrices

def crop_hough_circle(image, ellipse_coord):
    print('now i print the coordinates of the ellipse')
    print(ellipse_coord)

    if (len(ellipse_coord) == 0):
        rows = image.shape[0]
        im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        im_gray = cv2.medianBlur(im_gray, 5)

        circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT, 1, rows,
                                   param1=80, param2=20,
                                   minRadius=150, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(im_gray, (i[0], i[1]), i[2], (0, 255, 0), 4)
                # draw the center of the circle
                cv2.circle(im_gray, (i[0], i[1]), 2, (0, 128, 255), -1)
        else:
            print('circle not found')
            return None

        center_x = circles[0, 0, 0]
        center_y = circles[0, 0, 1]
        radius = circles[0, 0, 2]

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, (255,255,255), -1)
        result = cv2.bitwise_and(image, mask)
        #result = image * mask

    else:
        center = ellipse_coord[0]
        print(center)
        axes = ellipse_coord[1]
        print(axes)

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255,255,255), -1)
        result = cv2.bitwise_and(image, mask)

    return result, mask

def panorama_reconstruction(list_H_matrices, path, ellipse_coord):
    black_canvas = np.zeros((2000, 2000, 3), dtype="uint8")

    list_images = os.listdir(path)
    list_images.sort()

    panorama = np.copy(black_canvas)

    #parameters for the video
    size = black_canvas.shape[1], black_canvas.shape[0]
    fps = 25
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (500,500), True)

    for name in list_images:
        if('frame' not in name):
            list_images.remove(name)


    for index, image_name in enumerate(list_images):
        image = cv2.imread(os.path.join(os.getcwd(),path+image_name),1)
        print(image_name)
        image_circle, mask_circle = crop_hough_circle(image, ellipse_coord)


        erosion_size = 10
        erosion_shape = cv2.MORPH_ELLIPSE

        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                           (erosion_size, erosion_size))

        erosion = cv2.erode(mask_circle, element)

        canvas_center = np.array([[np.cos(0), np.sin(0), black_canvas.shape[1]/2], [-np.sin(0), np.cos(0), black_canvas.shape[0]/2], [0, 0, 1]])

        #H4p_orig = np.array([[0, 0],
        #                     [image_circle.shape[1], 0],
        #                     [0, image_circle.shape[0]],
        #                     [image_circle.shape[1], image_circle.shape[0]]])


        #x_orig = [H4p_orig[0][0], H4p_orig[1][0], H4p_orig[2][0], H4p_orig[3][0]]
        #y_orig = [H4p_orig[0][1], H4p_orig[1][1], H4p_orig[2][1], H4p_orig[3][1]]

        if (index == 0):
            img_origin = np.array([[np.cos(0), np.sin(0), -image_circle.shape[1]/2], [-np.sin(0), np.cos(0), -image_circle.shape[0]/2], [0, 0, 1]])
            #H4p = np.array([[black_canvas.shape[1]/2 - image_circle.shape[1]/2, black_canvas.shape[0]/2 - image_circle.shape[0]/2],
            #                        [black_canvas.shape[1]/2 + image_circle.shape[1]/2, black_canvas.shape[0]/2 - image_circle.shape[0]/2],
            #                        [black_canvas.shape[1]/2 - image_circle.shape[1]/2, black_canvas.shape[0]/2 + image_circle.shape[0]/2],
            #                        [black_canvas.shape[1]/2 + image_circle.shape[1]/2, black_canvas.shape[0]/2 + image_circle.shape[0]/2]])

            #H4p_hat = np.array([[360, 640],
            #                [360, 640],
            #                [360, 640],
            #                [360, 640]
            #])

            #H4p_prev = np.array([[0,0],
            #                     [image_circle.shape[1], 0],
            #                     [0, image_circle.shape[0]],
            #                     [image_circle.shape[1], image_circle.shape[0]]])
            H4p = canvas_center
            H4p = img_origin @ canvas_center
            print(image_circle.shape)



        else:

            tx, ty, R = list_H_matrices[index-1]
            H4p_rot = np.array(
                [[np.cos(R), np.sin(R),
                  ((1 - np.cos(R)) * (image_circle.shape[1] / 2) - (np.sin(R) * (image_circle.shape[0] / 2)))],
                 [-np.sin(R), np.cos(R),
                  (np.sin(R) * (image_circle.shape[1] / 2)) + ((1 - np.cos(R)) * (image_circle.shape[0] / 2))],
                 [0, 0, 1]])

            H4p_trasl = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
            H4p_hat = H4p_trasl @ H4p_rot
            H4p = H4p_prev @ H4p_hat
            #H4p_curr = list_H_matrices[index]
            print(H4p)
        H4p_prev = H4p
        #x_prev = [H4p_prev[0][0], H4p_prev[1][0], H4p_prev[2][0], H4p_prev[3][0]]
        #y_prev = [H4p_prev[0][1], H4p_prev[1][1], H4p_prev[2][1], H4p_prev[3][1]]


        #x = [H4p[0][0], H4p[1][0], H4p[2][0], H4p[3][0]]
        #y = [H4p[0][1], H4p[1][1], H4p[2][1], H4p[3][1]]

        #H_warp = DLT(H4p_orig, x_prev, y_prev)
        #H_warp = cv2.getPerspectiveTransform(np.float32(H4p_orig), np.float32(H4p))
        #K = np.array([1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0])
        #rot, tran, norm = cv2.decomposeHomographyMat(H_warp, K)

        #H_warp[0][0] = 1
        #H_warp[1][1] = 1
        #print(H_warp)
        #print(H_warp.shape)
        #tx, ty, R = 0., 0., 90. * np.pi / 180.


        panorama_curr = cv2.warpPerspective(image_circle, np.float32(H4p), (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
        panorama_y_cr_cb = cv2.cvtColor(panorama_curr, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(panorama_y_cr_cb)
        # Applying equalize Hist operation on Y channel.
        y_eq = cv2.equalizeHist(y)
        panorama_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
        panorama_curr = cv2.cvtColor(panorama_y_cr_cb_eq, cv2.COLOR_YCR_CB2RGB)

        mask_copy = cv2.warpPerspective(erosion, np.float32(H4p), (panorama.shape[1], panorama.shape[0]), flags=cv2.INTER_NEAREST)
        np.copyto(panorama, panorama_curr, where=mask_copy.astype(bool))

        edges = cv2.Canny(mask_copy, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) #edges with 3 channels
        result = cv2.bitwise_and(panorama_curr, edges)
        #
        dilatation_size = 3
        dilation_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                               (dilatation_size, dilatation_size))
        dilatation = cv2.dilate(edges, element)

        #dilatation_hsv = cv2.cvtColor(dilatation, cv2.COLOR_RGB2HSV)
        #dilatation_equalized = cv2.equalizeHist(dilatation_hsv)


        #dilatation_equalized = cv2.equalizeHist(dilatation)
        #plt.imshow(dilatation)
        #plt.show()

        panorama_with_border = np.copy(panorama)

        np.copyto(panorama_with_border, dilatation, where=dilatation.astype(bool))
        #panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        plt.imshow(panorama_with_border)
        # plt.show()
        #plt.imshow(edges)
        #plt.show()



        #panorama = cv2.rectangle(panorama, (x[0], y[0]), (x[3], y[3]), (255, 255, 255), 10)
        #plt.imshow(panorama)
        #plt.show()

        #pts = np.array([[x[0], y[0]], [x[1], y[1]], [x[3], y[3]], [x[2], y[2]]], np.int32)
        #pts = pts.reshape((-1, 1, 2))
        #panorama = cv2.polylines(panorama, [pts], True, (255, 255, 255), 2)
        panorama_with_border = cv2.resize(panorama_with_border, (500,500))
        cv2.imshow('', panorama_with_border)
        cv2.waitKey(5)


        print('this are the points before the update')
        #print(x_prev[0], y_prev[0])
        #print(x_prev[1], y_prev[1])
        #print(x_prev[3], y_prev[3])
        #print(x_prev[2], y_prev[2])

        print('this is H4p')
        print(H4p)

        H4p_prev = H4p

        #print('points after the update')
        #print(x[0], y[0])
        #print(x[1], y[1])
        #print(x[3], y[3])
        #print(x[2], y[2])


        #
        #showRaw = True
        #if showRaw:
        #    image_resized = cv2.resize(image, (round(image.shape[1]/2), round(image.shape[0]/2)), interpolation=cv2.INTER_LANCZOS4)
        #    panorama[:image_resized.shape[0], :image_resized.shape[1],:]=image_resized

        data = panorama_with_border
        out.write(data)


    #panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    #hsv_img = cv2.cvtColor(panorama, cv2.COLOR_RGB2HSV)
    #panorama_equalized = cv2.equalizeHist(hsv_img)
    #cv2.imwrite(os.path.join(os.getcwd(),'images/panorama_equalized.jpg'), panorama_equalized)
    #plt.imshow(panorama)
   # plt.show()
    out.release()

LOAD_IMAGES = False
if (LOAD_IMAGES == True):
    pre_processing(os.path.join(os.getcwd(),'frames from videos test/'), os.path.join(os.getcwd(),'processed test/'))

name = '2017_02_14_FRAMMENTO1'
list_H_matrices = prediction_matrix(os.path.join(os.getcwd(),'processed test/file_'+name+'.npy'))
coord_file = np.load('old files/folder_coord.npy', allow_pickle=True)

for elem in coord_file:
    print(elem[0])
    if (elem[0] == name):
        ellipse_coord = elem[1]

panorama_reconstruction(list_H_matrices, (os.path.join(os.getcwd(),'frames from videos test/frames from '+name+'/')), ellipse_coord)




