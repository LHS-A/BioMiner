# -- coding: utf-8 --
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from Segmentation_model.utils.spatial_transformation import Spatial_transformation 
from Segmentation_model.utils.util_tools import get_SDF_data

transform_tensor = transforms.ToTensor() 

def read_datasets(mode, data_path, image_folder="image"):
    
    images = []
    images_name = []

    if mode == "train":    
        image_folder_path = os.path.join(data_path, 'train', image_folder)
    elif mode == "val":
        image_folder_path = os.path.join(data_path, 'val', image_folder)
    elif mode == "test":
        image_folder_path = os.path.join(data_path, 'test', image_folder)

    images_name = os.listdir(image_folder_path)
    
    for name in images_name:
        img_path = os.path.join(image_folder_path, name)
        images.append(img_path)
        
    return images, images_name

class MyDataset(Dataset):
    def __init__(self, data_path, mode="train", image_folder="image"):
        self.mode = mode
        self.data_path = data_path
        self.image_folder = image_folder
        self.images, self.images_name = read_datasets(self.mode, data_path, image_folder)

    def __getitem__(self, index):
        image_path = self.images[index]
        image_name = self.images_name[index] 
    
        image = cv2.imread(image_path)

        if len(image.shape) == 2: # (H,W,C) 
            image = image[:,:,np.newaxis] 
            image = np.repeat(image,3,axis=-1) #（H,W,3)

        image_SDF = np.zeros_like(image[:,:,0]) # cause this is stage1，no SDF image and label are used.
        label = np.zeros_like(image[:,:,0])
        
        image = transform_tensor(image)
        image_SDF = transform_tensor(image_SDF)
        label = transform_tensor(label)
          
        return image, image_SDF, label, image_name 

    def __len__(self):
        return len(self.images)
    
class SingleImageDataset(Dataset):
    def __init__(self, data_path, image_folder="image"):
        self.data_path = data_path
        self.image_folder = image_folder
        self.image_dir = os.path.join(data_path, image_folder)
        self.images_name = os.listdir(self.image_dir)
        self.images = [os.path.join(self.image_dir, name) for name in self.images_name]

    def __getitem__(self, index):
        image_path = self.images[index]
        image_name = self.images_name[index] 
    
        image = cv2.imread(image_path)

        if len(image.shape) == 2:
            image = image[:,:,np.newaxis] 
            image = np.repeat(image,3,axis=-1)

        image_SDF = np.zeros_like(image[:,:,0])  
        label = np.zeros_like(image[:,:,0])
        
        image = transform_tensor(image)
        image_SDF = transform_tensor(image_SDF)
        label = transform_tensor(label)
          
        return image, image_SDF, label, image_name 

    def __len__(self):
        return len(self.images)

class SingleDataLoader():
    def __init__(self):
        pass

    def load_single_data(self, data_path, batch_size, image_folder="image"):
        dataset = SingleImageDataset(data_path, image_folder=image_folder)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
        return loader
    
class Data_loader():
    def __init__(self):
        pass

    def load_train_data(self, data_path, batch_size, image_folder="image"):
        dataset = MyDataset(data_path, mode="train", image_folder=image_folder)
        train_loader = DataLoader(dataset, batch_size, shuffle=False, pin_memory=False)
        return train_loader
    
    def load_val_data(self, data_path, batch_size, image_folder="image"):
        dataset = MyDataset(data_path, mode="val", image_folder=image_folder)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
        return val_loader
    
    def load_test_data(self, data_path, batch_size, image_folder="image"):
        dataset = MyDataset(data_path, mode="test", image_folder=image_folder)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
        return test_loader