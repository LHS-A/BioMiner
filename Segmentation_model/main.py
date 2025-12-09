# -- coding: utf-8 --
import sys
sys.path.append(r"/data/Desktop/BioMiner") 
import torch.nn as nn     
import torch.optim as optim
from Segmentation_model.utils.visualizer_tool import Visualizer
import random
from config import * 
args = Params() 
from Segmentation_model.utils.util_tools import *
from Segmentation_model.losses.BO_Loss import DiceLoss
from Segmentation_model.dataloader.Seg_loader import *
from train import train
from val import val
from test import test 
import warnings
import time 
import os 
import glob
import cv2
import numpy as np
warnings.filterwarnings("ignore")
import ssl  
ssl._create_default_https_context = ssl._create_unverified_context

seed = 111
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed)  

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("============================{}===========================".format(device))
vis = Visualizer(env=args.env_name, port=args.vis_port)

# Create two models: one for inference, one for training
# ================================== Morph-Net =====================================
# seg_model = setup_seg_model(device, args=args)
# train_model, optimizer = setup_train_model(device, pretrained=True)
# ================================== UNet =====================================
seg_model = setup_unet_seg_model(device, args=args)
train_model, optimizer = setup_unet_model(device, in_channels=3, out_channels=1)

# Setup data loaders
data_loader = Data_loader()
val_loader = data_loader.load_val_data(args, batch_size=args.val_batch)
test_loader = data_loader.load_test_data(args, batch_size=args.test_batch)
train_loader = data_loader.load_train_data(args, batch_size=args.train_batch)

# Setup loss functions
criterion = {
    "BCEloss": nn.BCEWithLogitsLoss().to(device),
    "DiceLoss": DiceLoss(args.num_classes).to(device),
    "CEloss": nn.CrossEntropyLoss().to(device)
}

# Setup scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.99, patience=5, verbose=False, 
    threshold=0.0001, threshold_mode="rel", cooldown=0, min_lr=1e-6, eps=1e-08
)

# ========================================== Train Model ==============================================

start = time.time() 
print("Start training MorphoBoost Model!")

# Initialize predict label folders with all-black images
initialize_predict_labels(args)

start_epoch = 0 
start_epoch = setup_training_resume(args, train_model, optimizer)

if args.resume_training == True:
    update_predict_labels(train_model, device, 100, args)

# Flag to track if seg_model weights have been loaded
seg_model_weights_loaded = False
args.save_pred = False
args.enhance_mode = "train"

for epoch in range(start_epoch, args.epochs): 
    args.epoch = epoch       
    
    # Load seg_model optimal weights when epoch > args.switch_epoch (only once)
    if epoch > args.switch_epoch and not seg_model_weights_loaded:
        args.enhance_mode = None
        print("================================================================")
        print("Epoch {} > args.switch_epoch: Loading optimal weights for seg_model".format(epoch))
        print("================================================================")
        seg_model = load_latest_seg_model_weights(seg_model, args.best_model_path)
        seg_model_weights_loaded = True
    
    # Train with two models
    train(args, vis, device, train_loader, seg_model, train_model, optimizer, criterion, epoch)
    
    # Validation with two models
    loss_val = val(args, vis, device, val_loader, seg_model, train_model, criterion, epoch)
    
    # Test with two models
    dice_pred, sen_pred = test(args, vis, seg_model, train_model, device, test_loader, criterion, epoch)

    if dice_pred > args.best_dice:  
        args.best_dice = dice_pred
        
        # Save only train_model
        best_model_path = os.path.join(args.best_model_path, f"best_model_{epoch}.pkl")
        torch.save(train_model.state_dict(), best_model_path)
        delete_previous_items(args.best_model_path)
        
        # Save checkpoint with only train_model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': train_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': args.best_dice,
            'scheduler_state_dict': scheduler.state_dict(),
        }
        checkpoint_path = os.path.join(args.checkpoint_dir_path, f"best_checkpoint_{epoch}.pkl")
        torch.save(checkpoint, checkpoint_path)
        delete_previous_items(args.checkpoint_dir_path)
        
        if epoch > args.switch_epoch:
            args.save_pred = True
            print(f"Epoch {epoch} > args.switch_epoch: save the best pred maps.")
            dice_pred, sen_pred = test(args, vis, seg_model, train_model, device, test_loader, criterion, epoch)
            args.save_pred = False

        if epoch < args.switch_epoch: # Only stage 1, we update the predict label.
            # Update predict labels after 25 epochs
            update_predict_labels(train_model, device, epoch, args)

end = time.time()
training_time = end - start
print(f"Training completed! Total time: {training_time:.2f} seconds")

