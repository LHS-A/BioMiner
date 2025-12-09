# -- coding: utf-8 --
import torch
import os
import shutil
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import glob
from config import * 
args = Params()  
import torch.nn.functional as F 
import glob
from typing import List, Tuple
from PIL import Image
import torchvision.transforms as transforms
from Segmentation_model.model.Morph_Net import create_morphonet_model
from Segmentation_model.model.UNet import create_unet_model

def save_best_model_predictions(args, seg_model, train_model, device, test_loader):

    seg_model.eval()
    train_model.eval()
    
    with torch.no_grad():
        for image_lst, SDF_image_lst, label_lst, image_name_lst in test_loader:
            image = image_lst.float().to(device)
            SDF_image = SDF_image_lst.float().to(device)
            label = label_lst.float().to(device)
            
            if args.epoch > args.switch_epoch:
                initial_seg = seg_model(image)
                enhanced_input = generate_enhanced_input(initial_seg, SDF_image)
                final_seg = train_model(enhanced_input)
                final_seg = final_seg + initial_seg
            else:
                final_seg = train_model(image)
            
            Predict_save(args, image, final_seg, label, "test", image_name_lst, len(image_name_lst))

def Predict_save(args, image_batch, pred_batch, pred_label_batch, mode, img_name, num_batch):
    pred_batch = (pred_batch.detach().cpu().numpy() * 255).astype(np.uint8) 
    pred_label_batch = (pred_label_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    image_batch = (image_batch.detach().cpu().numpy() * 255).astype(np.uint8)

    for i in range(num_batch):
        image = image_batch[i, :, :, :]
        pred_multi = pred_batch[i, :, :, :]
        pred_label_multi = pred_label_batch[i, :, :, :]

        name = img_name[i].split(".")[0] 
        for j in range(pred_batch.shape[1]):
            _, pred = cv2.threshold(pred_multi[j,:,:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(os.path.join(args.paper_save_cell, name + ".png"), pred)

def model_installization(model,weight): 
    model_pretrain_dict = torch.load(weight)
    model_dict = model.state_dict()
    # model_pretrain_dict = {k.replace("module.","module.online_encoder."):v for k,v in model_pretrain_dict.items()}  # if name has more or less n..you can replace and this is not data,you need not update!
    model_pretrain_dict = {k:v for k,v in model_pretrain_dict.items() if k in model_dict}  # get the same layer! if your model do not has pretrain module,then it will be random installization!
    model_dict.update(model_pretrain_dict) # update model_dict
    model.load_state_dict(model_dict)
    #freeze model!
    for name,param in model.named_parameters():
        if name in model_pretrain_dict:   
            param.requires_grad = False 
            # if name.split(".")[0] == "inc" or name.split(".")[0] == "nerve_out":
            #     param.requires_grad = True 
        else:
            # print(name)
            param.requires_grad = True
    
    return model    

def load_model_weights_only(model, checkpoint_path, args=None):
    """
    Load model weights only for continued training
    
    Args:
        model: model to be loaded
        checkpoint_path: path to weight file
        args: argument object (optional, for updating best_dice etc.)
    
    Returns:
        model: model with loaded weights
        start_epoch: suggested starting epoch (parsed from filename)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Weight file not found: {checkpoint_path}")
    
    print(f"Loading model weights: {checkpoint_path}")
    
    # load weights
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    
    # parse epoch from filename
    filename = os.path.basename(checkpoint_path)
    try:
        # extract 50 from "best_model_50.pkl"
        epoch_num = int(filename.split('_')[-1].split('.')[0])
        start_epoch = epoch_num + 1  # start from next epoch
    except:
        print("Cannot parse epoch from filename, will start from epoch 1")
        start_epoch = 1
    
    print(f"Model weights loaded successfully!")
    print(f"Detected model from epoch: {epoch_num}")
    print(f"Suggested to continue training from epoch: {start_epoch}")
    
    return model, start_epoch

def delete_previous_models(folder_path):
    files = os.listdir(folder_path)
    model_files = [file for file in files if file.endswith('.pkl') or file.endswith('.pth')]
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    if len(model_files) > 1:
        file_to_delete = model_files[0]
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)
        

def delete_previous_checkpoints(folder_path, keep_current=None):
    """
    Delete all model/checkpoint files in folder, keep only the latest one
    
    Args:
        folder_path: folder path
        keep_current: current file path to keep (optional)
    """
    if not os.path.exists(folder_path):
        return
        
    # get all pkl and pth files in folder
    pattern = os.path.join(folder_path, "*")
    all_files = glob.glob(pattern)
    model_files = [f for f in all_files if f.endswith('.pkl') or f.endswith('.pth')]
    
    if not model_files:
        return
    
    # sort by modification time, newest last
    model_files.sort(key=os.path.getmtime)
    
    # if specific file to keep, ensure it is in list
    if keep_current and keep_current in model_files:
        files_to_delete = [f for f in model_files if f != keep_current]
    else:
        # if no file specified, keep the newest one
        files_to_delete = model_files[:-1] if len(model_files) > 1 else []
    
    # delete all old files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted old file: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to delete file {file_path}: {e}")
    
    # print kept file info
    remaining_files = [f for f in model_files if f not in files_to_delete]
    if remaining_files:
        print(f"Kept files: {[os.path.basename(f) for f in remaining_files]}")
  
def setup_training_resume(args, S_model, optimizer_S):
    """
    Setup training resume state
    
    Args:
        args: argument object
        S_model: student model
        optimizer_S: student model optimizer
    
    Returns:
        start_epoch: epoch to start training
    """
    start_epoch = 0
    
    if args.resume_training:
        latest_checkpoint = args.S_checkpoint_dir_path_last
        if latest_checkpoint:
            try:
                start_epoch = load_checkpoint_for_resume(
                    latest_checkpoint, S_model, optimizer_S, args
                )
                print(f"Resumed training from checkpoint: {latest_checkpoint}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch...")
                start_epoch = 0
        else:
            print("Checkpoint file not found, starting training from scratch...")
    else:
        print("Starting training from scratch...")
    
    return start_epoch

def load_latest_seg_model_weights(seg_model, model_path):
    """
    Load the latest seg_model weights from the specified path
    """
    # Get all .pkl files in the model path
    model_files = glob.glob(os.path.join(model_path, "*.pkl"))
    
    if not model_files:
        print("Warning: No model files found in {}".format(model_path))
        return seg_model
    
    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model_path = model_files[0]
    
    print("Loading latest seg_model weights from: {}".format(latest_model_path))
    
    try:
        # Load the state dict
        state_dict = torch.load(latest_model_path, map_location="cuda")
        seg_model.load_state_dict(state_dict)
        print("Successfully loaded seg_model weights!")
    except Exception as e:
        print("Error loading seg_model weights: {}".format(e))
    
    return seg_model

def load_checkpoint_for_resume(checkpoint_path, model, optimizer, args):
    """
    Load checkpoint to resume training
    
    Args:
        checkpoint_path: checkpoint path
        model: student model
        optimizer: optimizer
        args: argument object
    
    Returns:
        start_epoch: epoch to start
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # load optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # restore best dice score
    args.best_dice = checkpoint['best_dice']
    
    # start from next epoch
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Checkpoint loaded successfully!")
    print(f"Resumed epoch: {checkpoint['epoch']}")
    print(f"Best dice score: {args.best_dice}")
    print(f"Continue training from epoch {start_epoch}")
    
    return start_epoch
   
def save_best_model(args,dice_pred,sen_pred,Train_Down_model):

    if  dice_pred > args.best_dice and sen_pred > args.sen_thed:
        args.best_dice = dice_pred
        args.best_sen = sen_pred
        print("best_dice:{},best_sen:{}".format(args.best_dice,args.best_sen))
        if Train_Down_model == True:
            move_file(args.Down_model_path,args.Down_best_model_path,'lhs_epoch_{}.pkl'.format(args.Down_epoch))
            move_file(args.Down_optimizer_path,args.Down_best_optimizer_path,'lhs_optimizer_{}.pth'.format(args.Down_epoch))
            delete_previous_items(args.Down_best_model_path)
            delete_previous_items(args.Down_best_optimizer_path)
        else:
            move_file(args.Up_model_path,args.Up_best_model_path,'lhs_epoch_{}.pkl'.format(args.Up_epoch))
            move_file(args.Up_optimizer_path,args.Up_best_optimizer_path,'lhs_optimizer_{}.pth'.format(args.Up_epoch))
            delete_previous_items(args.Up_best_model_path)
            delete_previous_items(args.Up_best_optimizer_path)

def delete_previous_items(folder_path):

    files = os.listdir(folder_path)
    model_files = [file for file in files if file.endswith('.pkl') or file.endswith('.pth')]

    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    if len(model_files) > 4:
        file_to_delete = model_files[0]
        file_path = os.path.join(folder_path, file_to_delete)
        os.remove(file_path)
 
def move_file(source_folder, target_folder, file_name):
    print("Perform move file!")
    source_path = os.path.join(source_folder,file_name)
    target_path = os.path.join(target_folder,file_name)
    shutil.copy(source_path, target_path)

def clear_folder(folder_path):
    files = os.listdir(folder_path)
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)

def rename(folder_path):
    files = os.listdir(folder_path)
    
    for file in files:
        os.rename(file,"1_"+file)


def resize(image,label,img_size):

    resized_image = cv2.resize(image, img_size)
    resized_label = cv2.resize(label, img_size)
    _, resized_label = cv2.threshold(resized_label, 128, 255, cv2.THRESH_BINARY) #将所有非255的像素全部设置为0;

    return resized_image,resized_label


def center_crop(image, label, new_h, new_w):
    h, w, _ = image.shape

    top = (h - new_h) // 2
    left = (w - new_w) // 2
    bottom = top + new_h
    right = left + new_w

    cropped_image = image[top:bottom, left:right, :]
    cropped_label = label[top:bottom, left:right, :]

    return cropped_image, cropped_label

def dialated_plain(image,label,dialate_pixels):
    """
    The dilated pixels must have a dilation label value between 0 and 1, and the result after multiplying by the original image must be an unsigned integer.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dialate_pixels, dialate_pixels))

    dia_label = cv2.dilate(label, kernel)
    dia_label = dia_label // 255
    dia_label = dia_label[:,:,np.newaxis] #(H,W,1)
    dia_image = (image * dia_label).astype(np.uint8)
    
    return dia_image
 
def get_latest_best_model(args):
    """
    Get the latest best model from args.best_model_path
    """
    if not os.path.exists(args.best_model_path):
        return None
    
    model_files = [f for f in os.listdir(args.best_model_path) if f.startswith('best_model_') and f.endswith('.pkl')]
    if not model_files:
        return None
    
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_model = os.path.join(args.best_model_path, model_files[-1])
    return latest_model

def setup_seg_model(device, args=None):
    """
    Setup segmentation model for inference only (no gradient update)
    Load the latest best model from args.best_model_path, or randomly initialize if not available
    """
    seg_model = create_morphonet_model(in_channels=3, out_channels=1, pretrained=False)
    
    latest_model_path = get_latest_best_model(args)
    if latest_model_path and os.path.exists(latest_model_path):
        seg_model_path = latest_model_path
        print(f"Loading latest best model: {seg_model_path}")
        seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
    else:
        print("No existing model found, using random initialization")
    
    seg_model.to(device)
    seg_model.eval()
    
    return seg_model

def generate_enhanced_input(initial_seg, original_image):
    """
    Generate enhanced input by concatenating initial segmentation with original image
    """
    # Ensure initial_seg has same spatial dimensions
    if initial_seg.shape[2:] != original_image.shape[2:]:
        initial_seg_resized = F.interpolate(initial_seg, size=original_image.shape[2:], 
                                          mode='bilinear', align_corners=True)
    else:
        initial_seg_resized = initial_seg
    
    # Convert 3-channel image to 2-channel by taking first two channels
    two_channel_image = original_image[:, :2, :, :]
    
    # Concatenate to form 3-channel input: [channel1, channel2, segmentation]
    enhanced_input = torch.cat([two_channel_image, initial_seg_resized], dim=1)
    
    return enhanced_input

def setup_train_model(device, pretrained=True):
    """
    Setup training model for parameter updates
    """
    train_model = create_morphonet_model(in_channels=3, out_channels=1, pretrained=pretrained)
    train_model.to(device)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)
    
    return train_model, optimizer

def setup_unet_seg_model(device, args=None):
    """
    Setup UNet segmentation model for inference only (no gradient update)
    Load the latest best model from checkpoint path, or randomly initialize if not available
    """
    seg_model = create_unet_model(in_channels=3, out_channels=1)
    
    # 假设最佳模型保存在固定的路径，你可以根据实际情况修改
    checkpoint_path = args.checkpoint_dir_path
    if checkpoint_path and os.path.exists(checkpoint_path):
        # 获取最新的模型文件
        model_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            latest_model_path = os.path.join(checkpoint_path, latest_model)
            print(f"Loading latest UNet model: {latest_model_path}")
            seg_model.load_state_dict(torch.load(latest_model_path, map_location=device))
        else:
            print("No model files found in checkpoint directory, using random initialization")
    else:
        print("Checkpoint directory not found, using random initialization")
    
    seg_model.to(device)
    seg_model.eval()
    
    return seg_model

def setup_unet_model(device, in_channels=3, out_channels=1):
    """
    Setup UNet model for training with optimizer
    """
    train_model = create_unet_model(in_channels=in_channels, out_channels=out_channels)
    train_model.to(device)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)
    
    return train_model, optimizer

def update_predict_labels(model, device, epoch, args):
    """
    Update predict labels for train, val, test sets using the current best model
    After 25 epochs, accumulate predictions using cv2.add
    """
    if epoch < 21:
        return
    
    print(f"Updating predict labels at epoch {epoch}")
    model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Process train, val, test sets
    for dataset_type in ['train', 'val', 'test']:
        if dataset_type == 'train':
            image_dir = os.path.join(args.data_path, 'train', args.image_folder)
            output_dir = args.train_predict_data_path
        elif dataset_type == 'val':
            image_dir = os.path.join(args.data_path, 'val', args.image_folder) 
            output_dir = args.val_predict_data_path
        else:
            image_dir = os.path.join(args.data_path, 'test', args.image_folder)
            output_dir = args.test_predict_data_path
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
        
        with torch.no_grad():
            for image_file in image_files:
                # Load and preprocess image
                image = Image.open(image_file).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get prediction
                pred = torch.sigmoid(model(image_tensor))
                pred_binary = (pred > 0.5).float()
                
                # Convert to numpy and scale to 0-255
                pred_np = (pred_binary.cpu().squeeze().numpy() * 255).astype(np.uint8)
                
                # Build output path
                base_name = os.path.basename(image_file)
                output_path = os.path.join(output_dir, base_name)
                
                # If file exists and epoch >= 20, use cv2.add to accumulate
                if os.path.exists(output_path):
                    old_pred = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
                    # Use cv2.add to accumulate predictions
                    combined = cv2.add(old_pred, pred_np)
                    # Ensure binary (0 or 255)
                    combined_binary = (combined > 0).astype(np.uint8) * 255
                    combined_binary = preprocess(combined_binary) # remove small pixel noise
                    cv2.imwrite(output_path, combined_binary)
                else:
                    # Save binary prediction (0 or 255)
                    pred_np = preprocess(pred_np) # remove small pixel noise
                    cv2.imwrite(output_path, pred_np) 

def initialize_predict_labels(args):
    """
    Initialize Predict_label folders with all-black images for train, val, test sets
    """
    print("Initializing Predict_label folders with all-black images...")
    
    for dataset_type in ['train', 'val', 'test']:
        if dataset_type == 'train':
            image_dir = os.path.join(args.data_path, 'train', args.image_folder)
            output_dir = args.train_predict_data_path
        elif dataset_type == 'val':
            image_dir = os.path.join(args.data_path, 'val', args.image_folder)
            output_dir = args.val_predict_data_path
        else:
            image_dir = os.path.join(args.data_path, 'test', args.image_folder)
            output_dir = args.test_predict_data_path
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
        
        for image_file in image_files:
            # Read original image to get dimensions
            original_image = cv2.imread(image_file)
            if original_image is None:
                print(f"Warning: Could not read image {image_file}, skipping...")
                continue
                
            # Get image dimensions (height, width)
            h, w = original_image.shape[:2]
            
            # Create all-black image with same dimensions
            black_image = np.zeros((h, w), dtype=np.uint16)  # 使用uint16为后续累积做准备
            
            # Build output path
            base_name = os.path.basename(image_file)
            output_path = os.path.join(output_dir, base_name)
            
            # Save all-black image
            cv2.imwrite(output_path, black_image)
            print(f"Created initial black image: {output_path}")
    
    print("Predict_label folders initialization completed!")
    
def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]

    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                sdf[sdf < 0] = 0
                # print(np.unique(sdf))
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def get_SDF_data(image,label,beta = 3):
    image = np.transpose(image, (2,0,1)) #[C,H,W]
    image = image[np.newaxis,:,:,:] # [1,C,H,W]
    label = label.squeeze()
    label = label[np.newaxis,np.newaxis,:,:] # [1,1,H,W]
    SDF_label = compute_sdf(label, label.shape)    
    SDF_label = np.exp(-beta*SDF_label) 
    SDF_image = (image * SDF_label).astype(np.uint8).squeeze()# [1,C,H,W] - [C,H,W] 
    SDF_label = SDF_label.squeeze() * 255
    SDF_label = SDF_label.astype(np.uint8)
    if len(SDF_image.shape) == 3:
        SDF_image = np.transpose(SDF_image, (1,2,0)) #[C,H,W] -> [H,W,C] 

    return SDF_image

def preprocess(pred_image):
    _, pred_components = cv2.connectedComponents(pred_image)
    num_pred_fibers = np.max(pred_components)

    label_lengths = []
    for fiber_label in range(1, num_pred_fibers + 1):
        fiber_mask = np.uint8(pred_components == fiber_label)
        fiber_length = np.sum(fiber_mask)
        label_lengths.append(fiber_length)

    pred_new = np.zeros_like(pred_image)   
    for fiber_label in range(1, num_pred_fibers + 1):
        fiber_mask = np.uint8(pred_components == fiber_label)
        fiber_length = np.sum(fiber_mask)
        if fiber_length < 21:
            fiber_mask[fiber_mask > 0] = 0
        pred_new += fiber_mask
        
    pred_new = pred_new * 255

    return pred_new      

def center_crop_and_pad(image, label,random_label,th, tw):
    h, w = image.shape[:2]
    x1 = max(0, int((w - tw) / 2))
    y1 = max(0, int((h - th) / 2))
    
    x2 = min(w, x1 + tw)
    y2 = min(h, y1 + th)
    
    random_label = np.resize(random_label, (h, w))
    
    if len(image.shape) == 2:
        cropped_image = np.zeros((th, tw), dtype=image.dtype)
    else:
        cropped_image = np.zeros((th, tw, image.shape[2]), dtype=image.dtype)
        
    cropped_label =  np.zeros((th, tw), dtype=label.dtype)
    cropped_random_label = np.zeros((th, tw), dtype=random_label.dtype)  

    cropped_image[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = image[y1:y2, x1:x2]
    cropped_label[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = label[y1:y2, x1:x2]
    cropped_random_label[(th - (y2 - y1)) // 2:(th - (y2 - y1)) // 2 + (y2 - y1),
                  (tw - (x2 - x1)) // 2:(tw - (x2 - x1)) // 2 + (x2 - x1)] = random_label[y1:y2, x1:x2]
    
    return cropped_image,cropped_label,cropped_random_label


def calculate_mean_and_std(metric):
    metric = np.array(metric)
    # print(metric.shape)
    # print("metric.shape:{}".format(metric.shape))
    # Per class metric of mean±std.
    mean_metric = np.mean(metric,axis=0)
    # print(mean_metric)
    std_metric = np.std(metric,axis=0)
    # Total metric of mean±std.
    mean_total = np.mean(mean_metric,axis=0)
    std_total = np.std(std_metric,axis=0)

    return mean_total,std_total,mean_metric, std_metric

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

def create_nonzero_mask(data):
    assert len(data.shape) == 3, "data must have shape (H, W, C)"
    nonzero_mask = np.zeros(data.shape[:2], dtype=bool)
    for c in range(data.shape[-1]):
        this_mask = data[..., c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minxidx = int(np.min(mask_voxel_coords[0]))
    maxxidx = int(np.max(mask_voxel_coords[0])) + 1
    minyidx = int(np.min(mask_voxel_coords[1]))
    maxyidx = int(np.max(mask_voxel_coords[1])) + 1
    return [minxidx, maxxidx], [minyidx, maxyidx]

def crop_to_bbox(image, bbox):
    return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

def random_crop_to_size(image, mask, size):
    h, w = image.shape[:2]
    crop_h, crop_w = size
    if h == crop_h and w == crop_w:
        return image, mask
    else:
        x = np.random.randint(0, h - crop_h + 1)
        y = np.random.randint(0, w - crop_w + 1)
        return image[y:y+crop_h, x:x+crop_w], mask[y:y+crop_h, x:x+crop_w]

def crop_to_nonzero(data, seg, roi_size, nonzero_label=-1):
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    max_x = bbox[0][1] - roi_size[1]
    max_y = bbox[1][1] - roi_size[0]
    if max_x < 0 or max_y < 0:
        raise ValueError("ROI size is larger than the nonzero region.")
    x = np.random.randint(0, max_x + 1)
    y = np.random.randint(0, max_y + 1)

    cropped_image, cropped_mask = random_crop_to_size(data, seg, roi_size)

    return cropped_image, cropped_mask
