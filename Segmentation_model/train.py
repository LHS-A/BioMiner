# -- coding: utf-8 --
import torch
import torch.nn.functional as F
from Segmentation_model.utils.util_tools import *
from Segmentation_model.losses.BO_Loss import calculate_bo_loss
from Segmentation_model.utils.util_tools import generate_enhanced_input

def train(args, vis, device, train_loader, seg_model, train_model, optimizer, criterion, epoch):
    """
    Training function with two models: 
    - seg_model: inference only for segmentation (no gradient update)
    - train_model: trains on concatenated segmentation results and SDF images (with gradient update)
    Stage control:
    - Epoch <= args.switch_epoch: Only use original image for training
    - Epoch > args.switch_epoch: Load best weights for seg_model (inference only) and enable fine-tuning stage
    """
    seg_model.eval()
    train_model.train()
    
    print("================================ Train {} epoch =====================================".format(epoch))
    
    for image_lst, SDF_image_lst, label_lst, image_name_lst in train_loader:
        image = image_lst.float().to(device)
        SDF_image = SDF_image_lst.float().to(device)
        label = label_lst.float().to(device)

        optimizer.zero_grad()

        # Stage control: Different processing based on epoch number
        if epoch <= args.switch_epoch:
            # Stage 1: Only use original image for training (first args.switch_epoch epochs)
            # Directly use image as input to train_model
            final_seg = train_model(image)
        else:
            # Stage 2: Enable fine-tuning with seg_model inference (after args.switch_epoch epochs)
            # Load best weights for seg_model (this should be done outside the training loop)
            with torch.no_grad():
                initial_seg = seg_model(image)
            
            # Generate enhanced input and get final segmentation
            enhanced_input = generate_enhanced_input(initial_seg, SDF_image)
            final_seg = train_model(enhanced_input)
            final_seg = final_seg + initial_seg
            
        # Calculate boundary optimization loss
        loss = calculate_bo_loss(final_seg, label, image, lambda_param=0.2, gamma=1.0)

        # Plot loss
        vis.plot(win="train_loss", y=loss.item(), con_point=len(args.train_loss),
                opts=dict(title="train_loss", xlabel="batch", ylabel="train_loss"))

        loss.backward()
        optimizer.step()