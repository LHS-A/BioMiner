# ============================ Note that this file is used for clinical downstream task validation. The file tree structure can be referenced from CORN-LCs. Only the args parameters need to be modified to run ===============================
# This script is used to generate prediction labels for the grading task, then calculate quantitative metrics based on these labels, and finally generate text inputs for the generative model based on the metrics.

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
from Segmentation_model.model.UNet import create_unet_model 


class Args:
    """Configuration class for parameters"""
    def __init__(self):
        # Data path configuration
        self.data_path = r'/data/Desktop/BioMiner/Dataset/Segmentation_task/CORN-1'
        self.image_folder = "image"
        self.save_folder = "final_pred" 
        # Model configuration
        self.mode_metric = "nerve"

args = Args()

def setup_label_folders(data_path, image_folder="image"):
    """
    Create label folder structure
    Copy the image folder structure to the label folder
    """
    print("Setting up label folder structure...")
    
    for dataset_type in ['train', 'val', 'test']:
        # Image folder path
        image_dir = os.path.join(data_path, dataset_type, image_folder)
        # Label folder path
        label_dir = os.path.join(data_path, dataset_type, args.save_folder)
         
        # If the label folder already exists, delete it and recreate it
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
            print(f"Deleted existing label folder: {label_dir}")
        
        # Copy the image folder structure to the label folder
        shutil.copytree(image_dir, label_dir)
        print(f"Created label folder: {label_dir}")


def save_predictions(pred_batch, image_names, mode, data_path):
    """
    Save model prediction results as binary images
    """
    # Convert prediction results to numpy arrays and scale to 0-255
    pred_batch = (pred_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    
    for i in range(len(image_names)):
        # Get single-channel prediction result
        pred = pred_batch[i, 0, :, :]
        
        # Binarize the prediction
        _, pred_binary = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
        
        # Keep the original file name
        name = os.path.basename(image_names[i])
        
        # Choose save path based on mode
        save_dir = os.path.join(data_path, mode, args.save_folder) 
        
        # Save prediction result
        save_path = os.path.join(save_dir, name)

        # Remove low connectivity domains of 25 pixels
        pred_binary = preprocess(pred_binary) 

        cv2.imwrite(save_path, pred_binary)
        print(f"Saved prediction result: {save_path}")


def calculate_test_metrics(pred_batch, label_batch, mode_metric, image_names, batch_size):
    """
    Calculate evaluation metrics for the test set
    """
    # Use batch_metrics_pred to calculate various metrics
    sen_batch, dice_batch, pre_batch, FDR_batch, MHD_batch = batch_metrics_pred(
        None, None, None, torch.sigmoid(pred_batch), label_batch, mode_metric, 
        image_names, batch_size
    )
    
    return {
        'sensitivity': sen_batch,
        'dice': dice_batch,
        'precision': pre_batch,
        'fdr': FDR_batch,
        'mhd': MHD_batch
    }


def generate_predictions(device, data_loader, model, data_path, mode="test", epoch=0):
    """
    Main function to generate and save prediction results
    mode: "train", "val", or "test"
    """
    print(f"==================== Generating {mode.upper()} set predictions for Epoch {epoch} ====================")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Process data based on mode
            if mode == "test":
                # Test set includes label data for metric calculation
                image_lst, SDF_image_lst, label_lst, image_name_lst = batch
            else:
                # Training and validation sets only need image data
                image_lst, SDF_image_lst, _, image_name_lst = batch
                
            # Move data to device
            image = image_lst.float().to(device)
            
            # First-stage inference: predict directly using the image
            pred = model(image)
            
            # Apply sigmoid and binarize
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            
            # Save prediction results
            save_predictions(pred_binary, image_name_lst, mode, data_path)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches...")


def calculate_test_set_metrics(device, test_loader, model, mode_metric="nerve", epoch=0):
    """
    Calculate evaluation metrics for the test set
    """
    print(f"==================== Calculating test set metrics for Epoch {epoch} ====================")
    
    # Initialize metric lists
    batch_sen_pred, batch_dice_pred, batch_pre_pred, batch_fdr_pred, batch_mhd_pred = [], [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (image_lst, SDF_image_lst, label_lst, image_name_lst) in enumerate(test_loader):
            # Move data to device
            image = image_lst.float().to(device)
            label = label_lst.float().to(device)
            
            # First-stage inference
            pred = model(image)
            
            # Calculate metrics
            metrics = calculate_test_metrics(pred, label, mode_metric, image_name_lst, len(image_name_lst))
            
            # Collect batch metrics
            batch_sen_pred.append(metrics['sensitivity'])
            batch_dice_pred.append(metrics['dice'])
            batch_pre_pred.append(metrics['precision'])
            batch_fdr_pred.append(metrics['fdr'])
            batch_mhd_pred.append(metrics['mhd'])
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"Calculated metrics for {batch_idx} batches...")
    
    # Calculate average metrics
    if batch_dice_pred:
        sen_pred, sen_pred_std, sen_percls_mean, sen_percls_std = calculate_mean_and_std(batch_sen_pred)
        dice_pred, dice_pred_std, dice_percls_mean, dice_percls_std = calculate_mean_and_std(batch_dice_pred)
        pre_pred, pre_pred_std, pre_percls_mean, pre_percls_std = calculate_mean_and_std(batch_pre_pred)
        fdr_pred, fdr_pred_std, fdr_percls_mean, fdr_percls_std = calculate_mean_and_std(batch_fdr_pred)
        mhd_pred, mhd_pred_std, mhd_percls_mean, mhd_percls_std = calculate_mean_and_std(batch_mhd_pred)
        
        print(f"==================== Epoch:{epoch} Test Set Metric Results ====================")
        print(f"Sensitivity: {sen_percls_mean:.4f}±{sen_percls_std:.4f}")
        print(f"Dice Coefficient: {dice_percls_mean:.4f}±{dice_percls_std:.4f}")
        print(f"Precision: {pre_percls_mean:.4f}±{pre_percls_std:.4f}")
        print(f"False Detection Rate: {fdr_percls_mean:.4f}±{fdr_percls_std:.4f}")
        print(f"Hausdorff Distance: {mhd_percls_mean:.4f}±{mhd_percls_std:.4f}")
        
        return dice_pred, sen_pred
    
    print("Warning: No valid metrics calculated")
    return None, None


def run_complete_prediction_pipeline(device, model, data_loader, data_path, mode_metric="nerve", epoch=0, setup_folders=True, image_folder="image"):
    """
    Complete prediction pipeline:
    1. Set up label folders (optional)
    2. Generate predictions for train/val/test sets
    3. Calculate test set metrics
    """
    print("Starting the complete prediction pipeline...")
    
    # Step 1: Set up label folders
    if setup_folders:
        setup_label_folders(data_path, image_folder)
    
    # Step 2: Generate predictions for all datasets
    print("Starting prediction generation...")
    generate_predictions(device, data_loader['train'], model, data_path, mode="train", epoch=epoch)
    generate_predictions(device, data_loader['val'], model, data_path, mode="val", epoch=epoch)
    generate_predictions(device, data_loader['test'], model, data_path, mode="test", epoch=epoch)
    
    # Step 3: Calculate test set metrics
    print("Starting test set metric calculation...")
    dice_pred, sen_pred = calculate_test_set_metrics(device, data_loader['test'], model, mode_metric, epoch=epoch)
    
    return dice_pred, sen_pred


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
    print("Loading data...")
    from Segmentation_model.dataloader.Create_label_loader import Data_loader
    data_loader = Data_loader()

    # Prepare all data loaders
    loaders = {
        'train': data_loader.load_train_data(data_path=args.data_path, batch_size=test_batch, image_folder=args.image_folder),
        'val': data_loader.load_val_data(data_path=args.data_path, batch_size=test_batch, image_folder=args.image_folder),
        'test': data_loader.load_test_data(data_path=args.data_path, batch_size=test_batch, image_folder=args.image_folder)
    }
    
    # Load model weights
    print("Loading model weights...")
    model_path = r"/data/Desktop/BioMiner/Segmentation_model/best_model/BioMiner_2025MIA_CORN-ProUNet_600/stage1_best_model_197.pkl"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights: {model_path}")
    else:
        print(f"Warning: Model weight file does not exist: {model_path}")
        print("Using randomly initialized model")
    
    # Run complete prediction pipeline
    print("Starting the complete prediction pipeline...")
    dice_pred, sen_pred = run_complete_prediction_pipeline(
        device=device, 
        model=model, 
        data_loader=loaders, 
        data_path=args.data_path,
        mode_metric=args.mode_metric,
        epoch=100, 
        setup_folders=True,
        image_folder=args.image_folder
    )
    
    # Output final results
    if dice_pred is not None and sen_pred is not None:
        print(f"Prediction pipeline completed - Dice Coefficient: {dice_pred:.4f}, Sensitivity: {sen_pred:.4f}")
    else:
        print("Prediction pipeline completed, but no valid metrics were obtained")
