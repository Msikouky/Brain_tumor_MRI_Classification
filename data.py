import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

# LOADING THE DATA

def load_data(dir_path, img_size=(100, 100)):
    X=[]
    y=[]
    i=0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
      if not path.startswith('.'):
        labels[i] = path
        # print(dir_path + '/' + path)
        for file in os.listdir(dir_path + '/' + path):
            if not file.startswith('.'):
               img = cv2.imread(dir_path + '/' + path + '/' + file)
              # print(dir_path + '/' + path + '/' + file)
               X.append(img)
               y.append(i)
        i += 1
    X = np.array(X)
    y = np.array(y)
    # print(f'{labels}')
    # print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels

# Train_data_path = "Splitted_data_set/TRAIN"
# Test_data_path = "Splitted_data_set/TEST"
# Val_data_path = "Splitted_data_set/VAL"
# IMG_SIZE = (224, 224)

# X_train, y_train, labels = load_data(Train_data_path, IMG_SIZE)
# X_test, y_test, _ = load_data(Test_data_path, IMG_SIZE)
# X_val, y_val, _ = load_data(Val_data_path, IMG_SIZE)

# DISTRIBUTION OF CLASSES AMONG SETS

def distribution_of_sets(trainSet, ValidationSet, testSet):
    y =dict()
    y[0] = []
    y[1] = []
    for set_name in (trainSet, ValidationSet, testSet):
        y[0].append(np.sum(set_name == 0))
        y[1].append(np.sum(set_name == 1))
    
    trace0 = go.Bar(
       x=['Train Set', 'Validation Set', 'Test Set'],
       y=y[0],
       name='No',
       marker=dict(color='#33cc33'),
       opacity=0.7
    )

    trace1 = go.Bar(
       x=['Train Set', 'Validation Set', 'Test Set'],
       y=y[1],
       name='Yes',
       marker=dict(color='#ff3300'),
       opacity=0.7
    )

    data = [trace0, trace1]

    layout = go.Layout(
       title='Count of classes in each set',
       xaxis = {'title': 'Set'},
       yaxis = {'title': 'Count'}
    )

    fig = go.Figure(data, layout)

    
# distribution_of_sets(y_train, y_val, y_test)


# Griplot for desired number of images (n) from the specified set

def plot_samples(X, y, labels_dict, n=50):
   
    for index in range(len(labels_dict)):
        images = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(15,6))
        c = 1
        
        for image in images:
            plt.subplot(i,j,c)
            plt.imshow(image[0])
            plt.xticks([])
            plt.yticks([])
            c += 1
        
        plt.suptitle(f'Tumor: {labels_dict[index]}')
        plt.show()


# plot_samples(X_train, y_train, labels, 10)

# HISTOGRAM OF RATIO DISTRIBUTIONS (RATION = WIDTH/HEIGHT)

def ratio_distribution_histogram(X_train, X_test, X_val):

    RATIO_LIST = []
    for set in (X_train, X_test, X_val):
        for image in set:
            RATIO_LIST.append(image.shape[1]/image.shape[0])
    plt.hist(RATIO_LIST)
    plt.title('Distribution of Image Ratios')
    plt.xlabel('Ratio Value')
    plt.ylabel('Count')
    plt.show()

# ratio_distribution_histogram(X_train, X_test, X_val)


