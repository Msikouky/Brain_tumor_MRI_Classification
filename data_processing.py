import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from data import load_data, plot_samples
import torchvision.transforms as transforms

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# NORMALIZATION OF OUR DATA SETS

IMG_SIZE =(244, 244)
# First step: Crop the brain out of the images

preprocess = transforms.Compose([
   # transforms.Resize((224, 224)),  # Resize the image to the VGG16 input size
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normalize using VGG16's mean values
        std=[0.229, 0.224, 0.225]    # Normalize using VGG16's standard deviation values
    ),
])

def crop_images(set_name, add_pixels_value=0):
    
    new_set = []
    for image in set_name:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5),0)
    
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # add contour on the image
        img_cnt = cv2.drawContours(image.copy(), [c], -1, (0, 255, 255), 4)
    
        # add extreme points
        img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
        img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
        img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
        img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

        # crop
        ADD_PIXELS = add_pixels_value
        new_img = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        new_set.append(new_img)

    return np.array(new_set)

# SAVE NEW CROPPED IMAGES IN DIFERENT FOLDERS
def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
        i += 1


# Resize and preprocess the images
def preprocess_imgs(set_name, img_size):
    """
    Resize the image and apply VGG-16 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        img = preprocess(img)
        img = img.permute(1, 2, 0)
        set_new.append(img)

    return np.array(set_new)


Train_data_path = "Splitted_data_set/TRAIN"
Test_data_path = "Splitted_data_set/TEST"
Val_data_path = "Splitted_data_set/VAL"

X_train, y_train, labels = load_data(Train_data_path, IMG_SIZE)
X_test, y_test, _ = load_data(Test_data_path, IMG_SIZE)
X_val, y_val, _ = load_data(Val_data_path, IMG_SIZE)

X_train_crop = crop_images(set_name=X_train)
X_val_crop = crop_images(set_name=X_val)
X_test_crop = crop_images(set_name=X_test)

#plot_samples(X_train_crop, y_train, labels, 10)

create_dir('CROPPED_DATA/TRAIN_CROP/NO')
create_dir('CROPPED_DATA/TRAIN_CROP/YES')
create_dir('CROPPED_DATA/VAL_CROP/NO')
create_dir('CROPPED_DATA/VAL_CROP/YES')
create_dir('CROPPED_DATA/TEST_CROP/NO')
create_dir('CROPPED_DATA/TEST_CROP/YES')

save_new_images(X_train_crop, y_train, folder_name='CROPPED_DATA/TRAIN_CROP/')
save_new_images(X_val_crop, y_val, folder_name='CROPPED_DATA/VAL_CROP/')
save_new_images(X_test_crop, y_test, folder_name='CROPPED_DATA/TEST_CROP/')

X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)
plot_samples(X_train_prep, y_train, labels, 10)