# ============================ NOTE: This file is used for clinical downstream task validation. The file tree structure can refer to DED, DM, HSK, PCS. Just modify the args parameters to run. ===============================
# Used to generate prediction labels for grading tasks, then create quantitative metrics from prediction labels, and finally generate text inputs for the generative model.

# -- coding: utf-8 --
import sys
sys.path.append(r"/data/Desktop/BioMiner") 
import numpy as np
import torch
import torch.nn as nn 
import os
import cv2
import shutil
from Segmentation_model.utils.metrics import *
from Segmentation_model.utils.visualizer_tool import Visualizer
from Segmentation_model.utils.util_tools import calculate_mean_and_std, preprocess
from Segmentation_model.model.Morph_Net import create_morphonet_model
from Segmentation_model.model.UNet import create_unet_model 


class Args:
    """Configuration class for parameters"""
    def __init__(self):
        # Data path configuration - directly point to the path containing the image folder
        self.data_path = r'/data/Desktop/BioMiner/Dataset/Grading_task/Activation/Grading_dataset/DryEYE/3'
        self.image_folder = "image"
        self.label_folder = "Predict_label"
        # Model configuration
        self.mode_metric = "nerve"


def setup_label_folder(data_path, image_folder="image"):
    """
    Create the label folder structure
    Copy the image folder structure to the label folder
    """
    print("Creating label folder structure...")
    
    # Path to the image folder
    image_dir = os.path.join(data_path, image_folder)
    # Path to the label folder
    label_dir = os.path.join(data_path, 'Predict_label')
    
    # If the label folder already exists, delete it and recreate
    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
        print(f"Existing label folder deleted: {label_dir}")
    
    # Copy the image folder structure to the label folder
    shutil.copytree(image_dir, label_dir)
    print(f"Label folder created: {label_dir}")


def save_predictions(pred_batch, image_names, data_path):
    """
    Save model prediction results as binary images
    """
    # Convert prediction results to numpy arrays and scale to 0-255
    pred_batch = (pred_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    
    for i in range(len(image_names)):
        # Get single-channel prediction result
        pred = pred_batch[i, 0, :, :]
        
        # Binarize
        _, pred_binary = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
        
        # Keep the original file name
        name = os.path.basename(image_names[i])
        
        # Path to save labels
        save_dir = os.path.join(data_path, 'Predict_label')
        
        # Save prediction result
        save_path = os.path.join(save_dir, name)

        # Remove low connectivity domains of 25 pixels
        pred_binary = preprocess(pred_binary) 

        cv2.imwrite(save_path, pred_binary)
        print(f"Prediction result saved: {save_path}")


def generate_predictions(device, data_loader, model, data_path):
    """
    Main function to generate and save prediction results
    """
    print("==================== Generating Prediction Results ====================")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Only need image data - using test mode data structure
            image_lst, SDF_image_lst, _, image_name_lst = batch
                
            # Move data to device
            image = image_lst.float().to(device)
            
            # Inference: directly predict using the image
            pred = model(image)
            
            # Apply sigmoid and binarize
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            
            # Save prediction results
            save_predictions(pred_binary, image_name_lst, data_path)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches...")


def run_prediction_pipeline(device, model, data_loader, data_path, setup_folders=True, image_folder="image"):
    """
    Prediction pipeline:
    1. Set up label folder (optional)
    2. Generate prediction results
    """
    print("Starting the prediction pipeline...")
    
    # Step 1: Set up label folder
    if setup_folders:
        setup_label_folder(data_path, image_folder)
    
    # Step 2: Generate prediction results
    print("Generating prediction results...")
    generate_predictions(device, data_loader, model, data_path)
    
    print("Prediction pipeline completed!")


if __name__ == "__main__":
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_unet_model(in_channels=3, out_channels=1).to(device)
    
    # Set parameters
    test_batch = 1
    
    # Create parameter object
    args = Args()
    
    # Load data
    from Segmentation_model.dataloader.Create_label_loader import SingleDataLoader
    print("Loading data...")
    # Use the new single-image data loader
    single_data_loader = SingleDataLoader()
    loader = single_data_loader.load_single_data(data_path=args.data_path, batch_size=test_batch, image_folder=args.image_folder)    
    
    # Load model weights
    print("Loading model weights...")
    model_path = r"/data/Desktop/BioMiner/Segmentation_model/best_model/BioMiner_2025MIA_CORN-ProUNet_600/best_model_197.pkl"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded: {model_path}")
    else:
        print(f"Warning: Model weight file does not exist: {model_path}")
        print("Using model with random initialization")
    
    # Run prediction pipeline
    print("Starting the prediction pipeline...")
    run_prediction_pipeline(
        device=device, 
        model=model, 
        data_loader=loader, 
        data_path=args.data_path,
        setup_folders=True,
        image_folder=args.image_folder
    )
