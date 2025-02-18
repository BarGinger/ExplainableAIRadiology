import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import zipfile

class CheXpertDataset(Dataset):
    def __init__(self, dataframe, class_names, zip_path, transform=None):
        self.dataframe = dataframe
        self.class_names = class_names
        self.zip_path = zip_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        labels = self.dataframe.iloc[idx][self.class_names].values.astype('float32')
        labels = torch.tensor(labels, dtype=torch.float32)

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_img_path = img_path.replace("CheXpert-v1.0-small/", "")
            with zip_ref.open(zip_img_path) as img_file:
                image = Image.open(img_file).convert('RGB') # .convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, labels

# KeyError: "There is no item named 'CheXpert-v1.0-small/train/patient36856/study9/view1_frontal.jpg' in the archive"


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import os


# class CheXpertDataset(Dataset):
#     def __init__(self, dataframe, class_names, transform=None):
#         self.dataframe = dataframe
#         self.class_names = class_names
#         self.transform = transform #use this later on to resize images and pre-process if we need it

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         img_path = self.dataframe.iloc[idx]['path']

#         if os.path.exists('/chexpert_data'):
#           FOLDER = '/chexpert_data/'
#         elif os.path.exists('/content/drive/MyDrive'):
#           FOLDER = '/content/drive/MyDrive/CheXpert-v1.0-small/'
#         else:
#           FOLDER = "" # Or handle the case where the directory doesn't exist
#         img_path = FOLDER + img_path
#         image = Image.open(img_path).convert('RGB')  #rgb format
#         labels = self.dataframe.iloc[idx][self.class_names].values.astype('float32')  # astype float32 otherwise error
#         labels = torch.tensor(labels, dtype=torch.float32)

#         if self.transform:
#             image = self.transform(image)

#         return image, labels