"""
============================ Note: This file is used for clinical downstream task validation. The file tree structure can refer to DED, DM, HSK, PCS. 
Simply modify the args parameters to run the program ===============================
Classification training task workflow:
1. Before execution, be sure to first run inference_or_label.py or inference_or_label_IMAGE.py to obtain Predict_label.
2. Then train the grading task model, obtain the optimal weights, and run utils/interpretability_CAM.py and cnn_visualization/gradcam.py to get the optimal ROI_image and paper interpretation figures.
3. Load the optimal checkpoint to fine-tune the grading model and obtain the optimal grading model.
Classification label inference workflow:
1. First, run the optimal segmentation model to get Predict_label.
2. Then load the optimal first-stage classification model to get the optimal ROI_image.
3. Call Vision_grading_model/inference_or_label.py to obtain the inference labels.
4. Filter the ideal labels corresponding to the disease.
"""
# TODO Modify the dataset_name and weight_name in the args parameters to run the program

import sys
sys.path.append(r"/data/Desktop/BioMiner")  
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import os
import csv
from Vision_grading_model.dataloader.Label_generator_loader import SingleImageData
from utils.evaluation_metrics import get_metrix
from Vision_grading_model.model.vision_grading import vision_grading_model
from collections import Counter

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

args = {
    'dataset_name': "CORN-LCs/0",  # The folder where the dataset is stored
    'weight_name': "CORN-LCs/1_finetune",  # The folder where the pth is stored, not the specific path of the pth
    'pretrained': False  # Set according to the actual situation
}
base_path = f"/data/Desktop/BioMiner/Dataset/Grading_task/Activation/Grading_dataset/{args['dataset_name']}"
args['base_path'] = base_path 

args['root'] = base_path
args['pred_path'] = f"{base_path}" 

def write2txt(txt_name, seg_name, class_name):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(txt_name), exist_ok=True)
    
    # Create the file if it doesn't exist, or append to it if it does
    with open(txt_name, 'a', encoding='utf-8') as f:
        f.write(f"{seg_name}\t{class_name}\n")

def write_statistics_to_readme(counter, total_count, readme_path):
    """
    Write statistics to the readme.txt file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("Prediction Statistics:\n")
        f.write("=====================\n\n")
        for label in [0, 1, 2, 3]:
            count = counter[label]
            f.write(f"Level {label}: {count} images\n")
        f.write(f"Total: {total_count} images\n\n")
        
        # Add percentage information
        f.write("Percentage Distribution:\n")
        f.write("========================\n")
        for label in [0, 1, 2, 3]:
            count = counter[label]
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            f.write(f"Level {label}: {count} images ({percentage:.2f}%)\n")

def ensure_txt_file_exists(txt_path):
    """Ensure the txt file exists, create an empty file if it doesn't"""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    if not os.path.exists(txt_path):
        with open(txt_path, 'w', encoding='utf-8') as f:
            pass  # Create an empty file
        print(f"Created empty file: {txt_path}")

def load_net():
    # First, create a model instance
    net = vision_grading_model(num_class=4, model_name="resnet50", pre_train=args['pretrained']).cuda()

    # Load the saved weights
    last_path = sorted(Path(os.path.join("/data/Desktop/BioMiner/Vision_grading_model/best_weight", args['weight_name'])).iterdir(),
                       key=lambda p: p.stat().st_mtime)[-1]
    assert last_path.is_file(), "The newest entry is not a file."
    
    # Load the state dictionary
    state_dict = torch.load(str(last_path))
    net.load_state_dict(state_dict)
    
    return net

def predict_single_image_folder():
    """
    Predict the grading of all images in a single image folder
    """
    net = load_net()
    net.eval()
    
    # Load data - using the new SingleImageData class
    print("Loading image data from a single folder...")
    data = SingleImageData(image_dir=os.path.join(args['root'], 'image'))
    data_loader = DataLoader(data, batch_size=64, num_workers=8)
    
    print(f"Found {len(data)} images in the image folder")
    
    preds = []
    
    # Prediction result file path - save to image.txt in the image folder
    pred_txt_path = os.path.join(args['root'], 'image.txt')
    
    # Check if the file exists and delete it if it does
    if os.path.exists(pred_txt_path):
        os.remove(pred_txt_path)
        print(f"Removed existing file: {pred_txt_path}")
    
    # Create a new empty file
    with open(pred_txt_path, 'w', encoding='utf-8') as f:
        pass
    print(f"Created a new file for predictions: {pred_txt_path}")
    
    with torch.no_grad():
        print("Predicting images in the single folder...")
        for idx, batch in enumerate(data_loader):
            img = batch[0]["img"].cuda()
            seg = batch[0]["seg"].cuda()
            roi = batch[0]["roi"].cuda()
            sdf = batch[0]["sdf"].cuda()
            image = torch.cat((img, img, seg), dim=1)
            roi = torch.cat((roi, roi), dim=1)

            x1, roi, predictions = net(image, roi)
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            preds.extend(predictions)
            
            for i in range(len(predictions)):
                file_name = os.path.basename(batch[1]["img_name"][i])
                class_name = predictions[i]
                write2txt(pred_txt_path, file_name, class_name)
            
            # Print progress
            if idx % 10 == 0:
                print(f"Processed {idx} batches...")
    
    print(f"Predictions saved to: {pred_txt_path}")
    print(f"Total predictions made: {len(preds)}")
    
    # Statistics
    counter = Counter(preds)
    print("\nPrediction statistics:")
    for label in [0, 1, 2, 3]:
        count = counter[label]
        print(f"  Level {label}: {count} images")
    print(f"  Total: {len(preds)} images")
    
    # Write statistics to readme.txt
    readme_path = os.path.join(args['root'], 'readme.txt')
    write_statistics_to_readme(counter, len(preds), readme_path)
    print(f"Statistics saved to: {readme_path}")

def predict():
    """
    Main prediction function - predict a single image folder
    """
    print(f"\n{'='*50}")
    print("Processing a single image folder")
    print(f"{'='*50}")
    
    predict_single_image_folder()
    
    print(f"\n{'='*50}")
    print("Single folder prediction completed!")
    print(f"{'='*50}")

if __name__ == '__main__':
    predict()
