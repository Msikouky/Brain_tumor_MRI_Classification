import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import shutil
import random

# class BrainTumorDataSet(Dataset):
#     def __init__(self, path):

#         self.image_path = path.replace("\\", "/")

#     def __getitem__(self, index):
#         """ Reading image """
#         image = cv2.imread(self.path[index], cv2.IMREAD_COLOR)
#         image = image/255.0 ## (244, 244)
#         image = np.transpose(image, (1, 0))  ## (244, 244)
#         image = np.expand_dims(image, axis=0)
#         image = image.astype(np.float32)
#         image = torch.from_numpy(image)

#         return image
    
def split_Data(source_path, destination_path):
        source_path = source_path.replace("\\", "/")
        destination_path = destination_path.replace("\\", "/")
        # Define subfolder names (e.g., "yes" and "no")
        subfolders =["yes", "no"]

        # Define the percentage split for train, val, and test sets
        train_percent = 0.8
        val_percent = 0.15
        test_percent = 0.05

        # Create train, val, and test directories within the destination_root
        for split in ["TRAIN", "VAL", "TEST"]:
            split_dir = os.path.join(destination_path, split)
            os.makedirs(split_dir, exist_ok=True)

            # Create "yes" and "no" subdirectories within each split
            for subfolder in subfolders:
                subfolder_dir = os.path.join(split_dir, subfolder)
                os.makedirs(subfolder_dir, exist_ok=True)
            
        # Iterate through the "yes" and "no" subfolders
        for subfolder in subfolders:
            subfolder_path = os.path.join(source_path, subfolder)

            # Get the list of image filenames in the subfolder
            image_filenames = os.listdir(subfolder_path)
            random.shuffle(image_filenames)  # Shuffle the filenames randomly

            # Split the filenames into train, val, and test sets
            num_images = len(image_filenames)
            num_train = int(train_percent * num_images)
            num_val = int(val_percent * num_images)

            train_filenames = image_filenames[:num_train]
            val_filenames = image_filenames[num_train:num_train + num_val]
            test_filenames = image_filenames[num_train + num_val:]

            # Copy the images to the corresponding train, val, and test directories
            for split_filenames, split_dir in [(train_filenames, "TRAIN"), (val_filenames, "VAL"), (test_filenames, "TEST")]:
                for filename in split_filenames:
                    src_path = os.path.join(subfolder_path, filename)
                    dst_path = os.path.join(destination_path, split_dir, subfolder, filename)
                    shutil.copy(src_path, dst_path)
    
        print("Data split and folders created successfully.")


source_path = "brain_tumor_dataset/"
destination_path = "Splitted_data_set/"
spilted_data = split_Data(source_path, destination_path)