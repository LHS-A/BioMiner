"""
============================ Please note: This file is used for CORN-LCs validation, as their file tree structure is as follows (refer to CORN-LCs). Only modify the args parameters to run ===============================
Classification training task workflow:
1. Before execution, first run inference_or_label.py or inference_or_label_IMAGE.py to obtain Predict_label
2. Then train the grading task model, obtain the optimal weights, and execute utils/interpretability_CAM.py and cnn_visualization/gradcam.py to get the optimal ROI_image and paper interpretability figures
3. Load the optimal checkpoint and continue fine-tuning the grading model to obtain the optimal grading model.
Classification label inference workflow:
1. First execute the optimal segmentation model to obtain Predict_label
2. Then load the first-stage optimal classification model to obtain the optimal ROI_image
3. Then call Vision_grading_model/inference_or_label.py to obtain inference labels
4. Filter ideal labels corresponding to diseases
"""

import sys
sys.path.append(r"/data/Desktop/BioMiner")  
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import os
import csv
from Vision_grading_model.dataloader.Label_generator_loader import MyData
from utils.evaluation_metrics import get_metrix
from Vision_grading_model.model.vision_grading import vision_grading_model

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

args = { 
    'dataset_name' : "CORN-LCs/1", # The folder where the dataset is stored
    'weight_name': "CORN-LCs/1_finetune", # The folder where pth files are stored, not the specific pth path
    'pretrained': False  # Set according to actual situation
}
base_path = f"/data/Desktop/BioMiner/Dataset/Grading_task/Activation/Grading_dataset/{args['dataset_name']}"
args['base_path'] = base_path 

args['root'] = base_path
args['pred_path'] = f"{base_path}" 

def write2txt(txt_name, seg_name, class_name):
    # Ensure directory exists
    os.makedirs(os.path.dirname(txt_name), exist_ok=True)
    
    # If file doesn't exist, create it; if it exists, append to it
    with open(txt_name, 'a', encoding='utf-8') as f:
        f.write(f"{seg_name}\t{class_name}\n")

def ensure_txt_file_exists(txt_path):
    """Ensure txt file exists, create empty file if it doesn't"""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    if not os.path.exists(txt_path):
        with open(txt_path, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        print(f"Created empty file: {txt_path}")

def load_net():
    # First create model instance
    net = vision_grading_model(num_class=4, model_name="resnet50", pre_train=args['pretrained']).cuda()

    # Load saved weights
    last_path = sorted(Path(os.path.join("/data/Desktop/BioMiner/Vision_grading_model/best_weight", args['weight_name'])).iterdir(),
                       key=lambda p: p.stat().st_mtime)[-1]
    assert last_path.is_file(), "The newest entry is not a file."
    
    # Load state dictionary
    state_dict = torch.load(str(last_path))
    net.load_state_dict(state_dict)
    
    return net

def predict_split(split_name):
    net = load_net()
    net.eval()
    
    # Load data - now directly from directory, not dependent on txt files
    print(f"Loading {split_name} data from directory...")
    data = MyData(root_dir=args['root'], train=split_name)
    data_loader = DataLoader(data, batch_size=64, num_workers=8)
    
    print(f"Found {len(data)} samples in {split_name} set")
    
    preds, gts = [], []
    
    # Prediction results file path - directly saved to corresponding split folder
    pred_txt_path = os.path.join(args['root'], split_name, f"{split_name}.txt")
    
    # Check if file exists, delete if it does
    if os.path.exists(pred_txt_path):
        os.remove(pred_txt_path)
        print(f"Removed existing file: {pred_txt_path}")
    
    # Create new empty file
    with open(pred_txt_path, 'w', encoding='utf-8') as f:
        pass
    print(f"Created new file for predictions: {pred_txt_path}")
    
    with torch.no_grad():
        print(f"Predicting {split_name} data...")
        for idx, batch in enumerate(data_loader):
            img = batch[0]["img"].cuda()
            seg = batch[0]["seg"].cuda()
            roi = batch[0]["roi"].cuda()
            sdf = batch[0]["sdf"].cuda()
            image = torch.cat((img, img, seg), dim=1)
            roi = torch.cat((roi, roi), dim=1)
            label = batch[1]["img_id"].cuda()

            x1, roi, predictions = net(image, roi)
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            label = label.data.cpu().numpy()
            preds.extend(predictions)
            gts.extend(label)
            for i in range(len(predictions)):
                file_name = os.path.basename(batch[1]["img_name"][i])
                class_name = predictions[i]
                write2txt(pred_txt_path, file_name, class_name)
    
    print(f"Predictions for {split_name} saved to: {pred_txt_path}")
    print(f"Total predictions made: {len(preds)}")
    
    # preds = np.array(preds)
    # gts = np.array(gts)
    # wacc, wse, wsp = get_metrix(preds, gts)
    # print("wAcc: %.4f" % wacc[0])

def predict(Data_list):
    for split in Data_list:
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print(f"{'='*50}")
        predict_split(split)
    
    print(f"\n{'='*50}")
    print("All predictions completed!")
    print(f"{'='*50}")

if __name__ == '__main__':
    # Data_list=['train', 'val', 'test']
    Data_list=['test']
    predict(Data_list)
