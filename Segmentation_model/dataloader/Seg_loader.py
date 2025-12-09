# -- coding: utf-8 --
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from Segmentation_model.dataloader.augmentation import *
from Segmentation_model.utils.spatial_transformation import Spatial_transformation 
from Segmentation_model.utils.util_tools import get_SDF_data

transform_tensor = transforms.ToTensor() 
 
def read_datasets(mode, args):
    images = []
    labels = []
    predict_labels = [] 

    if mode == "train":    
        train_folder = os.path.join(args.data_path, 'train')
        image_folder = os.path.join(train_folder, args.image_folder)
        images_name = os.listdir(image_folder)
        label_folder = os.path.join(train_folder, args.label_folder)
        predict_folder = args.train_predict_data_path
        
    elif mode == "val":
        val_folder = os.path.join(args.data_path, 'val')
        image_folder = os.path.join(val_folder, args.image_folder)
        images_name = os.listdir(image_folder)
        label_folder = os.path.join(val_folder, args.label_folder)
        predict_folder = args.val_predict_data_path
          
    elif mode == "test":
        test_folder = os.path.join(args.data_path, 'test')
        image_folder = os.path.join(test_folder, args.image_folder)
        images_name = os.listdir(image_folder) 
        label_folder = os.path.join(test_folder, args.label_folder)
        predict_folder = args.test_predict_data_path

    for name in images_name:
        img_path = os.path.join(image_folder, name)
        images.append(img_path)
        label_path = os.path.join(label_folder, name.split(".")[0] + ".png")
        labels.append(label_path)        
        predict_label_path = os.path.join(predict_folder, name)
        predict_labels.append(predict_label_path)
        
    return images, labels, predict_labels, images_name

class MyDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.images, self.labels, self.predict_labels, self.images_name = read_datasets(self.mode, args)
        self.args = args

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        predict_label_path = self.predict_labels[index] 
        image_name = self.images_name[index] 
    
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        predict_label = cv2.imread(predict_label_path,0)

        if len(image.shape) == 2: # (H,W,C) 
            image = image[:,:,np.newaxis] 
            image = np.repeat(image,3,axis=-1) #（H,W,3）

        # image, label = Spatial_transformation(image, label)  
        label = label[:,:,np.newaxis] 
        predict_label = predict_label[:,:,np.newaxis] 
        
        # Not use in SDF_image
        # if self.mode == self.args.enhance_mode:
        #     image, label = apply_augmentations(image, label)

        image_SDF = get_SDF_data(image,predict_label)

        image = transform_tensor(image)
        image_SDF = transform_tensor(image_SDF)
        label = transform_tensor(label)
          
        return image, image_SDF, label, image_name 

    def __len__(self):
        assert len(self.images) == len(self.labels) == len(self.predict_labels)
        return len(self.images)

class Data_loader():
    def __init__(self):
        pass

    def load_train_data(self, args, batch_size):
        dataset = MyDataset(args, mode="train")
        train_loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)
        return train_loader
    
    def load_val_data(self, args, batch_size):
        dataset = MyDataset(args, mode="val")
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        return val_loader
    
    def load_test_data(self, args, batch_size):
        dataset = MyDataset(args, mode="test")
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        return test_loader