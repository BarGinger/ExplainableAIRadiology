# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import zipfile

# class CheXpertDataset(Dataset):
#     def __init__(self, dataframe, class_names, zip_path, transform=None):
#         self.dataframe = dataframe
#         self.class_names = class_names
#         self.zip_path = zip_path
#         self.transform = transform
#         self.images = []
#         self.labels = []

#         # Preload data into memory
#         with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
#             for idx in range(len(self.dataframe)):
#                 img_path = self.dataframe.iloc[idx]['path']
#                 labels = self.dataframe.iloc[idx][self.class_names].values.astype('float32')
#                 labels = torch.tensor(labels, dtype=torch.float32)

#                 zip_img_path = img_path.replace("CheXpert-v1.0-small/", "")
#                 with zip_ref.open(zip_img_path) as img_file:
#                     image = Image.open(img_file).convert('RGB')
#                     if self.transform:
#                         image = self.transform(image)
#                     self.images.append(image)
#                     self.labels.append(labels)

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class CheXpertDataset(Dataset):
    def __init__(self, dataframe, class_names, zip_path, transform=None):
        self.dataframe = dataframe
        self.class_names = class_names
        self.transform = transform #use this later on to resize images and pre-process if we need it

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']

        # if os.path.exists(f'/{zip_path}'):
        #   FOLDER = f'/{zip_path}/'
        # elif os.path.exists('/content/drive/MyDrive'):
        #   FOLDER = '/content/drive/MyDrive/CheXpert-v1.0-small/'
        # else:
        #   FOLDER = "" # Or handle the case where the directory doesn't exist
        FOLDER = ""
        img_path = FOLDER + img_path
        image = Image.open(img_path).convert('RGB')  #rgb format
        labels = self.dataframe.iloc[idx][self.class_names].values.astype('float32')  # astype float32 otherwise error
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels