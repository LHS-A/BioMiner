from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
import warnings 
import numpy as np
import numbers

warnings.filterwarnings('ignore')

def load_dataset_from_directory(root_dir, train="train"):
    """
    Load dataset paths directly from image directories without relying on TXT files
    Args:
        root_dir: Root directory containing data
        train: Dataset type - "train", "val", or "test"
    Returns:
        images_path: List of original image paths
        segs_path: List of segmentation image paths  
        rois_path: List of ROI image paths
        sdfs_path: List of SDF image paths
        img_names: List of image file names
    """
    images_path = []
    segs_path = []
    rois_path = []
    sdfs_path = []
    img_names = []
    
    # Define paths for different data types based on split
    if train == "train":  
        image_path = os.path.join(root_dir, "train", "image")
        predict_path = os.path.join(root_dir, "train", "Predict_label")
        ROI_path = os.path.join(root_dir, "train", "ROI_image")
        sdf_path = os.path.join(root_dir, "train", "SDF_image")
        print("image_path", image_path)
    elif train == "val":  
        image_path = os.path.join(root_dir, "val", "image")
        predict_path = os.path.join(root_dir, "val", "Predict_label")
        ROI_path = os.path.join(root_dir, "val", "ROI_image")
        sdf_path = os.path.join(root_dir, "val", "SDF_image")
    elif train == "test":
        image_path = os.path.join(root_dir, "test", "image")
        predict_path = os.path.join(root_dir, "test", "Predict_label")
        ROI_path = os.path.join(root_dir, "test", "ROI_image")
        sdf_path = os.path.join(root_dir, "test", "SDF_image")
    else:
        raise ValueError("train parameter must be 'train', 'val', or 'test'")
    
    # Check if directories exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image directory not found: {image_path}")
    if not os.path.exists(predict_path):
        raise FileNotFoundError(f"Predict_label directory not found: {predict_path}")
    if not os.path.exists(ROI_path):
        raise FileNotFoundError(f"ROI_image directory not found: {ROI_path}")
    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"SDF_image directory not found: {sdf_path}")
    
    # Get all image files from the image directory
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if len(image_files) == 0:
        print(f"Warning: No image files found in {image_path}")
        return images_path, segs_path, rois_path, sdfs_path, img_names
    
    print(f"Found {len(image_files)} images in {image_path}")
    
    # Construct full paths for all image types
    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)
        seg_path = os.path.join(predict_path, img_file)
        roi_path = os.path.join(ROI_path, img_file)
        sdf_path_full = os.path.join(sdf_path, img_file)
        
        # Check if all corresponding files exist
        if (os.path.exists(img_path) and os.path.exists(seg_path) and 
            os.path.exists(roi_path) and os.path.exists(sdf_path_full)):
            
            images_path.append(img_path)
            segs_path.append(seg_path)
            rois_path.append(roi_path)
            sdfs_path.append(sdf_path_full)
            img_names.append(img_file)
        else:
            print(f"Warning: Missing files for {img_file}")
            print(f"  Image: {os.path.exists(img_path)}")
            print(f"  Seg: {os.path.exists(seg_path)}")
            print(f"  ROI: {os.path.exists(roi_path)}")
            print(f"  SDF: {os.path.exists(sdf_path_full)}")
    
    print(f"Successfully loaded {len(images_path)} complete samples for {train}")
    return images_path, segs_path, rois_path, sdfs_path, img_names


class SingleImageData(Dataset):

    def __init__(self, image_dir, size=384):
        self.image_dir = image_dir
        self.resize = size
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No image files found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        self.roi_dir = os.path.join(os.path.dirname(image_dir), 'ROI_image')
        self.seg_dir = os.path.join(os.path.dirname(image_dir), 'Predict_label')
        
        if not os.path.exists(self.roi_dir):
            raise FileNotFoundError(f"ROI_image directory not found: {self.roi_dir}")
        if not os.path.exists(self.seg_dir):
            raise FileNotFoundError(f"Predict_label directory not found: {self.seg_dir}")
        
        self.transform_image = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        
        self.transform_roi = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        image = Image.open(image_path).convert("L")
        
        roi_path = os.path.join(self.roi_dir, image_name)
        roi = Image.open(roi_path).convert("L")
        
        seg_path = os.path.join(self.seg_dir, image_name)
        seg = Image.open(seg_path).convert("L")
        
        sdf_array = np.zeros_like(np.array(image))
        sdf = Image.fromarray(sdf_array).convert("L")
        
        image_tensor = self.transform_image(image)
        seg_tensor = self.transform_image(seg)
        roi_tensor = self.transform_roi(roi)
        sdf_tensor = self.transform_image(sdf)
        
        images = {
            "img": image_tensor,
            "seg": seg_tensor,
            "roi": roi_tensor,
            "sdf": sdf_tensor,
        }
        

        img_class = {
            "img_id": 0, 
            "img_name": image_path 
        }
        
        return images, img_class
  
class MyData(Dataset):
    def __init__(self,
                 root_dir,
                 train="train",
                 size=384):
        """
        Simplified data loader for inference - no augmentations needed
        """
        self.root_dir = root_dir
        self.train = train
        self.resize = size
        
        # Transformation pipelines - keep only what's needed for inference
        self.transform_image = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        
        self.transform_roi = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.339, std=0.138),
        ])
        
        # Load dataset paths directly from directories
        self.images_path, self.segs_path, self.roi_path, self.sdf_path, self.img_names = load_dataset_from_directory(self.root_dir, self.train)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        """Get single data sample for inference"""
        # Load image paths
        img_path = self.images_path[idx]      # Original image
        seg_path = self.segs_path[idx]        # Segmentation image
        roi_path = self.roi_path[idx]         # ROI image
        sdf_path = self.sdf_path[idx]         # SDF image
        img_name = self.img_names[idx]        # Image file name
        
        # Load all four image types
        try:
            image = Image.open(img_path).convert("L")    # Original image 
            seg = Image.open(seg_path).convert("L")      # Segmentation
            roi = Image.open(roi_path).convert("L")      # ROI
            sdf = Image.open(sdf_path).convert("L")      # SDF image
        except Exception as e:
            print(f"Error loading images for {img_path}: {e}")
            raise
        
        # Apply transformations (no augmentation for inference)
        image = self.transform_image(image)
        seg = self.transform_image(seg)
        roi = self.transform_roi(roi)
        sdf = self.transform_image(sdf)
        
        # Package all image types
        images = {
            "img": image,
            "seg": seg,
            "roi": roi,
            "sdf": sdf,
        }
        
        # For inference, we don't have true labels, so we use placeholder
        # The actual prediction will come from the model
        img_class = {
            "img_id": 0,  # Placeholder, not used in inference
            "img_name": img_path  # Keep full path for reference
        }
        
        return images, img_class
