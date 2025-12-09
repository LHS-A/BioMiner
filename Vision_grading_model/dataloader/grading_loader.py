from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
import random
import warnings 
import csv
import numpy as np
import numbers

warnings.filterwarnings('ignore')

def load_dataset(root_dir, train="train"):
    """
    Load dataset paths from TXT files
    Args:
        root_dir: Root directory containing TXT files
        train: Dataset type - "train", "val", or "test"
    Returns:
        images_path: List of original image paths
        segs_path: List of segmentation image paths  
        rois_path: List of ROI image paths
        sdfs_path: List of SDF image paths
        class_id: List of class labels
    """
    images_path = []
    segs_path = []
    rois_path = []
    sdfs_path = []
    class_id = []
    
    # Define paths for different data types based on split
    if train == "train":  
        image_path = root_dir + r"/train/image"  # Original images
        predict_path = root_dir + r"/train/Predict_label"  # Segmentation images
        ROI_path = root_dir + r"/train/ROI_image"  # ROI images
        sdf_path = root_dir + r"/train/SDF_image"  # SDF images

    elif train == "val":  
        image_path = root_dir + r"/val/image"
        predict_path = root_dir + r"/val/Predict_label"
        ROI_path = root_dir + r"/val/ROI_image"
        sdf_path = root_dir + r"/val/SDF_image"

    elif train == "test":
        image_path = root_dir + r"/test/image"
        predict_path = root_dir + r"/test/Predict_label"
        ROI_path = root_dir + r"/test/ROI_image"
        sdf_path = root_dir + r"/test/SDF_image"

    else:
        raise ValueError("train parameter must be 'train', 'val', or 'test'")
    
    # Read TXT file with tab delimiter and construct full paths for all image types
    txt_file_path = os.path.join(root_dir, train, train + ".txt")
    
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Split by tab character
            items = line.split('\t')
            
            # Check if we have exactly 2 items
            if len(items) != 2:
                print(f"Warning: Line {line_num} has incorrect format: {line}")
                continue
                
            img_name, id_str = items
            img_name = img_name.strip()
            id_str = id_str.strip()
            
            # Construct full paths for different data types
            img_path = os.path.join(image_path, img_name)      # Original image
            seg_path = os.path.join(predict_path, img_name)    # Segmentation
            roi_path = os.path.join(ROI_path, img_name)        # ROI
            sdf_path_full = os.path.join(sdf_path, img_name)   # SDF
            
            images_path.append(img_path)
            segs_path.append(seg_path)
            rois_path.append(roi_path)
            sdfs_path.append(sdf_path_full)
            class_id.append(int(id_str))
    
    # print(f"Successfully loaded {len(images_path)} samples from {txt_file_path}")
    
    return images_path, segs_path, rois_path, sdfs_path, class_id


class MyData(Dataset):
    def __init__(self,
                 root_dir,
                 train="train",
                 rotate=45,
                 flip=True,
                 random_crop=True,
                 size=384):

        self.root_dir = root_dir
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.resize = size
        self.random_crop = random_crop
        
        # Transformation pipelines
        self.transform_train = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        self.transform_roi = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        
        # Load dataset paths - now including SDF paths
        self.images_path, self.segs_path, self.roi_path, self.sdf_path, self.class_id = load_dataset(self.root_dir, self.train)

    def __len__(self):
        return len(self.images_path)

    def RandomCrop(self, img, seg, roi, sdf, crop_size):
        """Random crop augmentation for all image types"""
        if isinstance(crop_size, numbers.Number):
            crop_size = (int(crop_size), int(crop_size))
        else:
            crop_size = crop_size
        w, h = img.size
        th, tw = crop_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        img = F.crop(img, i, j, th, tw)
        seg = F.crop(seg, i, j, th, tw)
        roi = F.crop(roi, i, j, th, tw)
        sdf = F.crop(sdf, i, j, th, tw)
        return img, seg, roi, sdf

    def RandomHorizonalFlip(self, img, seg, roi, sdf, p):
        """Random horizontal flip augmentation"""
        if random.random() < p:
            img = F.hflip(img)
            seg = F.hflip(seg)
            roi = F.hflip(roi)
            sdf = F.hflip(sdf)
        return img, seg, roi, sdf

    def RandomVerticalFlip(self, img, seg, roi, sdf, p):
        """Random vertical flip augmentation"""
        if random.random() < p:
            img = F.vflip(img)
            seg = F.vflip(seg)
            roi = F.vflip(roi)
            sdf = F.vflip(sdf)
        return img, seg, roi, sdf

    def RandomRotation(self, img, seg, roi, sdf, degrees, p):
        """Random rotation augmentation"""
        if random.random() < p:
            angle = random.uniform(-degrees, degrees)
            img = F.rotate(img, angle)
            seg = F.rotate(seg, angle)
            roi = F.rotate(roi, angle)
            sdf = F.rotate(sdf, angle)
        return img, seg, roi, sdf

    def RandomColorJitter(self, img, seg, roi, sdf, p):
        """Random color jitter augmentation"""
        if random.random() < p:
            brightness = random.uniform(0.2, 0.8)
            contrast = random.uniform(0.2, 0.8)
            saturation = random.uniform(0.2, 0.8)
            img = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)(img)
            seg = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)(seg)
            roi = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)(roi)
            sdf = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)(sdf)
        return img, seg, roi, sdf

    def RandomSpNoise(self, img, seg, roi, sdf, p=0.25, mean=0, var=0.005):
        """Random salt and pepper noise augmentation"""
        if random.random() < p:
            img = np.asarray(img)
            seg = np.asarray(seg)
            roi = np.asarray(roi)
            sdf = np.asarray(sdf)
            img = np.array(img / 255, dtype=float)
            seg = np.array(seg / 255, dtype=float)
            roi = np.array(roi / 255, dtype=float)
            sdf = np.array(sdf / 255, dtype=float)
            noise = np.random.normal(mean, var ** 0.5, img.shape)
            img = img + noise
            seg = seg + noise
            # roi = roi + noise
            sdf = sdf + noise
            if img.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            img = np.clip(img, low_clip, 1.0)
            img = np.uint8(img * 255)
            img = Image.fromarray(np.uint8(img))
            seg = np.clip(seg, low_clip, 1.0)
            seg = np.uint8(seg * 255)
            seg = Image.fromarray(np.uint8(seg))
            roi = np.clip(roi, low_clip, 1.0)
            roi = np.uint8(roi * 255)
            roi = Image.fromarray(np.uint8(roi))
            sdf = np.clip(sdf, low_clip, 1.0)
            sdf = np.uint8(sdf * 255)
            sdf = Image.fromarray(np.uint8(sdf))
        return img, seg, roi, sdf

    def __getitem__(self, idx):
        """Get single data sample"""
        # Load image paths for all four types
        img_path = self.images_path[idx]      # Original image
        seg_path = self.segs_path[idx]        # Segmentation image
        roi_path = self.roi_path[idx]         # ROI image
        sdf_path = self.sdf_path[idx]         # SDF image
        img_id = self.class_id[idx]
        
        # Load all four image types
        image = Image.open(img_path).convert("L")    # Original image 
        seg = Image.open(seg_path).convert("L")      # Segmentation
        roi = Image.open(roi_path).convert("L")      # ROI
        sdf = Image.open(sdf_path).convert("L")      # SDF image

        # Apply data augmentation for training
        if self.train == "train":
            image, seg, roi, sdf = self.RandomSpNoise(image, seg, roi, sdf)
            image, seg, roi, sdf = self.RandomRotation(image, seg, roi, sdf, self.rotate, 0.6)
            image, seg, roi, sdf = self.RandomColorJitter(image, seg, roi, sdf, 0.25)
            image, seg, roi, sdf = self.RandomHorizonalFlip(image, seg, roi, sdf, 0.5)
            image, seg, roi, sdf = self.RandomVerticalFlip(image, seg, roi, sdf, 0.5)
            image, seg, roi, sdf = self.RandomCrop(image, seg, roi, sdf, crop_size=self.resize)
            image = self.transform_train(image)
            seg = self.transform_train(seg)
            roi = self.transform_roi(roi)
            sdf = self.transform_train(sdf)
        else:
            # Apply test transformations
            image = self.transform_test(image)
            seg = self.transform_test(seg)
            roi = self.transform_roi(roi)
            sdf = self.transform_test(sdf)
        
        # Package all image types
        images = {"img": image,
                  "seg": seg,
                  "roi": roi,
                  "sdf": sdf,
                  }
        img_class = {"img_id"  : img_id,
                     "img_name": self.images_path[idx]}
        return images, img_class