# -- coding: utf-8 --
import numpy as np
import torch
from Segmentation_model.utils.metrics import *
from Segmentation_model.utils.util_tools import calculate_mean_and_std
from config import *
from train import generate_enhanced_input

args = Params() 

def test(args, vis, seg_model, train_model, device, test_loader, criterion, epoch):
    batch_sen_pred, batch_dice_pred, batch_pre_pred, batch_fdr_pred, batch_mhd_pred = [], [], [], [], []

    seg_model.eval()  # seg_model always in eval mode for inference
    train_model.eval()  # train_model in eval mode for testing
    
    with torch.no_grad():
        for image_lst, SDF_image_lst, label_lst, image_name_lst in test_loader:
            image = image_lst.float().to(device)
            SDF_image = SDF_image_lst.float().to(device)
            label = label_lst.float().to(device)
            
            # Stage control: Different processing based on epoch number
            if epoch <= args.switch_epoch:
                # Stage 1: Only use original image for testing (first args.switch_epoch epochs)
                # Directly use image as input to train_model
                final_pred = train_model(image)
            else:
                # Stage 2: Enable fine-tuning with seg_model inference (after args.switch_epoch epochs)
                # Stage 1: seg_model generates initial segmentation
                initial_seg = seg_model(image)
                # Stage 2: train_model processes enhanced input (concatenated segmentation + SDF)
                enhanced_input = generate_enhanced_input(initial_seg, SDF_image)
                final_pred = train_model(enhanced_input)
                final_pred = final_pred + initial_seg
            # Calculate metrics using the final prediction
            sen_batch, dice_batch, pre_batch, FDR_batch, MHD_batch = batch_metrics_pred(
                args, vis, image, torch.sigmoid(final_pred), label, args.mode_metric, 
                image_name_lst, args.test_batch
            )      
            
            batch_sen_pred.append(sen_batch)
            batch_dice_pred.append(dice_batch)
            batch_pre_pred.append(pre_batch)
            batch_fdr_pred.append(FDR_batch)
            batch_mhd_pred.append(MHD_batch)
        
        # Calculate mean and std for all metrics
        sen_pred, sen_pred_std, sen_percls_mean, sen_percls_std = calculate_mean_and_std(batch_sen_pred)
        dice_pred, dice_pred_std, dice_percls_mean, dice_percls_std = calculate_mean_and_std(batch_dice_pred)
        pre_pred, pre_pred_std, pre_percls_mean, pre_percls_std = calculate_mean_and_std(batch_pre_pred)
        fdr_pred, fdr_pred_std, fdr_percls_mean, fdr_percls_std = calculate_mean_and_std(batch_fdr_pred)
        mhd_pred, mhd_pred_std, mhd_percls_mean, mhd_percls_std = calculate_mean_and_std(batch_mhd_pred)
        
        # Store metrics in args
        args.metric_test["total_sen_pred"].append(sen_pred)
        args.metric_test["total_dice_pred"].append(dice_pred)
        args.metric_test["total_pre_pred"].append(pre_pred)    
        args.metric_test["total_fdr_pred"].append(fdr_pred)
        args.metric_test["total_mhd_pred"].append(mhd_pred)

        # Plot metrics
        vis.plot_metrics_total(args.metrics_dict_test)
        
        print("================================ Epoch:{} Test Metric =====================================".format(epoch))
        print("sen_PerCls: {}±{}, dice_PerCls: {}±{}, pre_PerCls: {}±{}, fdr_PerCls: {}±{}, mhd_PerCls: {}±{}".format(
            sen_percls_mean, sen_percls_std, dice_percls_mean, dice_percls_std, 
            pre_percls_mean, pre_percls_std, fdr_percls_mean, fdr_percls_std, 
            mhd_percls_mean, mhd_percls_std))
        
    return dice_pred, sen_pred