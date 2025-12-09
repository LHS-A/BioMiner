# -- coding: utf-8 --
import sys
sys.path.append(r"/data/Desktop/BioMiner/") 
import numpy as np
import sklearn #pip install -U scikit-learn==0.22.1
from sklearn.metrics import roc_auc_score 
import cv2
import warnings
warnings.filterwarnings("ignore")
from Segmentation_model.utils.mhd import * 
from scipy import spatial
import os

def iou(a, b, epsilon=1e-5):
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)

    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)
    
    union = np.logical_or(a, b)
    union = np.sum(union)
  
    iou = intersection / (union + epsilon)
    
    return iou

def single_categories_count(pred, gt):

    FP = float(np.sum((pred == 255) & (gt != 255)))
    FN = float(np.sum((pred != 255) & (gt == 255)))
    TP = float(np.sum((pred == 255) & (gt == 255)))
    TN = float(np.sum((pred != 255) & (gt != 255)))

    return FP, FN, TP, TN

def get_single_indicator(pred, gt, mode):
    FP, FN, TP, TN = single_categories_count(pred, gt)
    if pred.sum() > 0 and gt.sum() > 0:          
        sen = (TP) / (TP + FN + 1e-10)
        dice = (2 * TP) / ((TP + FN) + (TP + FP) + 1e-10)
        FDR = FP / (FP + TP + 1e-10)
        # F1 = (2 * pre * sen) / (pre + sen + 1e-10) # F1 = dice,so they just need have one ,OK!
        pre = TP / (TP + FP + 1e-10)
        jaccard = (1 - spatial.distance.jaccard(pred.flatten(), gt.flatten()))
        # spe = TN / (TN + FP + 1e-10)
        # _, gt = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            AUC = sklearn.metrics.roc_auc_score((gt / 255.0).flatten(), (pred / 255.0).flatten())
        except Exception as e:
            AUC = 0
        
        IOU = iou(pred/255.0,gt/255.0)
        VOE = 1-IOU 
        MHD = mhd_d23(pred//255 , gt//255)
        return sen, dice, pre, FDR, MHD

    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 1, 1, 0, 0
    else:
        return 0, 0, 0, 1, 1
        
def batch_metrics_pred(args,vis,image_batch ,pred_batch, pred_label_batch, mode,img_name,num_batch):
    sen_batch, dice_batch, MHD_batch, FDR_batch, pre_batch=  [],[],[],[],[]

    pred_batch = (pred_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    pred_label_batch = (pred_label_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    image_batch = (image_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    for j in range(pred_batch.shape[1]):
        sen_batch.append([]);dice_batch.append([]);MHD_batch.append([]);FDR_batch.append([]);pre_batch.append([])  
    
    for i in range(num_batch):
        image = image_batch[i, :, :, :]
        pred_multi = pred_batch[i, :, :, :]
        pred_label_multi = pred_label_batch[i, :, :, :]
        
        name = img_name[i].split(".")[0]
        slices = []
        for j in range(pred_batch.shape[1]):
            if args.open_OTUS == True:
                _, pred = cv2.threshold(pred_multi[j,:,:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:   
                _, pred = cv2.threshold(pred_multi[j,:,:], 127, 255, cv2.THRESH_BINARY)
            slices.append(pred) 

            if args.save_pred == True:
                cv2.imwrite(os.path.join(args.paper_save_cell,name + ".png"), pred)

            sen, dice, pre, FDR, MHD = get_single_indicator(pred, pred_label_multi[j,:,:], mode)
            sen_batch[j].append(sen);dice_batch[j].append(dice);MHD_batch[j].append(MHD);FDR_batch[j].append(FDR);pre_batch[j].append(pre)
            vis.plot_pred_contrast(pred,pred_label_multi[j,:,:],image[j,:,:])                

        pred = np.stack(slices,axis=0) 

    sen_batch = np.nanmean(sen_batch, axis=1);dice_batch = np.nanmean(dice_batch, axis=1)
    MHD_batch = np.nanmean(MHD_batch, axis=1);FDR_batch = np.nanmean(FDR_batch, axis=1)
    pre_batch = np.nanmean(pre_batch, axis=1)
  
    return sen_batch, dice_batch, pre_batch, FDR_batch, MHD_batch 

