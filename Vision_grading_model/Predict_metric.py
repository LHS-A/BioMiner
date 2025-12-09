# The function evaluates model grading performance and saves all image prediction probabilities to vision_grading_predictions.json,
# Used for logistic regression multimodal fusion with text_grading_predictions.json.
# TODO Modify dataset_name and weight_name in args parameters to run the program
import sys
sys.path.append(r"/data/Desktop/BioMiner")   
import os
import torch
import json
from torch.utils.data import DataLoader
from pathlib import Path
from Vision_grading_model.dataloader.grading_loader import MyData
from Vision_grading_model.model.vision_grading import vision_grading_model
from utils.evaluation_metrics import get_metrix
from prettytable import PrettyTable
import numpy as np
import pandas as pd

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

args = {
    'dataset_name': "CORN-LCs/1", # The folder where the dataset is stored
    'weight_name': "CORN-LCs/1_finetune", # The folder where pth files are stored, not the specific pth path # BASELINE_finetune   CORN-LCs/1_finetune
    'pretrained': False,
    'batch_size': 64
}

base_path = f"/data/Desktop/BioMiner/Dataset/Grading_task/Activation/Grading_dataset/{args['dataset_name']}"
args['base_path'] = base_path

def load_net():
    """Load trained model"""
    net = vision_grading_model(num_class=4, model_name="resnet50", pre_train=args['pretrained']).cuda()

    # Load saved weights
    weight_dir = os.path.join("/data/Desktop/BioMiner/Vision_grading_model/best_weight", args['weight_name'])
    last_path = sorted(Path(weight_dir).iterdir(),
                       key=lambda p: p.stat().st_mtime)[-1]
    assert last_path.is_file(), "The newest entry is not a file."
    
    # Load state dictionary
    state_dict = torch.load(str(last_path))
    net.load_state_dict(state_dict)
    
    print(f"Model loaded successfully from: {last_path}")
    return net

def model_eval(net, iters):
    """Modified model_eval function to return image names and prediction probabilities"""
    print("Start testing model...")
    test_data = MyData(args['base_path'], train="test")
    batchs_data = DataLoader(test_data, batch_size=args['batch_size'], num_workers=8)
    pred = []
    pred_probs = []  # Store prediction probabilities
    target = []
    image_names = []
    net.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batchs_data):
            img = batch[0]["img"].cuda()
            seg = batch[0]["seg"].cuda()
            roi = batch[0]["roi"].cuda()
            sdf = batch[0]["sdf"].cuda()
            image = torch.cat((img, img, seg), dim=1)
            roi = torch.cat((roi, roi), dim=1)
            class_id = batch[1]["img_id"].cuda()
            x1, roi, predictions = net(image, roi)
            
            # Get prediction probabilities
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            probabilities_np = probabilities.data.cpu().numpy()
            pred_probs.extend(probabilities_np)
            
            # Get prediction labels
            predictions_labels = torch.argmax(predictions, dim=1)
            predictions_labels = predictions_labels.data.cpu().numpy()
            class_id = class_id.data.cpu().numpy()
            pred.extend(predictions_labels)
            target.extend(class_id)
            
            # Get current batch image names
            batch_image_names = batch[1]["img_name"]
            image_names.extend(batch_image_names)
            
    pred = np.asarray(pred)
    target = np.asarray(target)
    pred_probs = np.asarray(pred_probs)
    wacc, wse, wsp = get_metrix(pred, target)
    x_acc1, x_sen1, y_spe1 = iters, iters, iters
    y_acc1, y_sen1, y_spe1 = wacc[0], wse[0], wsp[0]
    
    metrics = {
        'mean_acc': wacc[0],
        'mean_se': wse[0], 
        'mean_sp': wsp[0],
        'class_acc': wacc[1] if len(wacc) > 1 else [wacc[0]],
        'class_se': wse[1] if len(wse) > 1 else [wse[0]],
        'class_sp': wsp[1] if len(wsp) > 1 else [wsp[0]]
    }
    
    table = PrettyTable()
    table.field_names = ["wAcc", "wSe", "wSp"]
    table.add_row(["{0:.4f}".format(y_acc1), "{0:.4f}".format(y_sen1), "{0:.4f}".format(y_spe1)])
    print(table)
    return wacc[0], metrics, pred, target, image_names, pred_probs

def save_predictions_to_json(image_names, pred_probs, true_labels, output_path):
    """Save prediction results to JSON file"""
    predictions_list = []
    
    for i in range(len(image_names)):
        # Get image filename (including extension)
        image_name_only = os.path.basename(image_names[i])
        
        # Convert probability values to Python list
        probabilities_list = pred_probs[i].tolist()
        
        # Build dictionary
        prediction_dict = {
            "name": image_name_only,
            "vision_prediction_probs": probabilities_list,
            "true_label": int(true_labels[i])  # Ensure it's the true label
        }
        
        predictions_list.append(prediction_dict)
    
    # Create output directory (if it doesn't exist)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_list, f, indent=2, ensure_ascii=False)
    
    print(f"Prediction results saved to: {output_path}")
    print(f"Saved {len(predictions_list)} prediction records")

def evaluate_model():
    """Main evaluation function"""
    print(f"{'='*60}")
    print("MODEL PERFORMANCE EVALUATION - TEST SET")
    print(f"{'='*60}")
    
    # Load model
    net = load_net()
    
    # Call model_eval function for test set inference, get prediction probabilities
    accuracy, metrics, pred, target, image_names, pred_probs = model_eval(net, iters=1)
    
    print(f"\n{'='*60}")
    print("DETAILED CLASS-WISE PERFORMANCE")
    print(f"{'='*60}")
    
    # Display detailed metrics for each class
    class_table = PrettyTable()
    class_table.field_names = ["Class", "Accuracy", "Sensitivity", "Specificity"]
    
    # Add metrics for each class
    for i in range(len(metrics['class_acc'])):
        class_table.add_row([
            f"Level {i}",
            f"{metrics['class_acc'][i]:.4f}",
            f"{metrics['class_se'][i]:.4f}", 
            f"{metrics['class_sp'][i]:.4f}"
        ])
    
    # Add mean row
    class_table.add_row([
        "MEAN",
        f"{metrics['mean_acc']:.4f}",
        f"{metrics['mean_se']:.4f}",
        f"{metrics['mean_sp']:.4f}"
    ])
    
    print(class_table)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Weighted Accuracy: {metrics['mean_acc']:.4f}")
    print(f"Overall Weighted Sensitivity: {metrics['mean_se']:.4f}")
    print(f"Overall Weighted Specificity: {metrics['mean_sp']:.4f}")
    
    # New feature: Save prediction results to JSON file
    print(f"\n{'='*60}")
    print("SAVING PREDICTION RESULTS TO JSON FILE")
    print(f"{'='*60}")
    
    json_output_path = "/data/Desktop/BioMiner/Vision_grading_model/vision_grading_predictions.json"
    save_predictions_to_json(image_names, pred_probs, target, json_output_path)
    
    # Original functionality: Statistics for images with true label 1 or 2 but prediction errors
    print(f"\n{'='*60}")
    print("EXPORTING PREDICTION ERRORS FOR CLASS 1 & 2")
    print(f"{'='*60}")
    
    # Create DataFrame for prediction errors
    error_data = []
    for i in range(len(target)):
        true_label = target[i]
        pred_label = pred[i]
        # Check condition: true label is 2 or 1, and prediction is incorrect
        if (true_label == 2 or true_label == 1) and true_label != pred_label:
            # Save only image filename (name + extension)
            image_name_only = os.path.basename(image_names[i])
            error_data.append({
                'image_name': image_name_only,
                'prediction': pred_label
            })
    
    # Save to CSV file, using \t as delimiter
    error_df = pd.DataFrame(error_data)
    csv_path = os.path.join(args['base_path'], 'prediction_errors.csv')
    error_df.to_csv(csv_path, sep='\t', index=False)
    
    print(f"Found {len(error_data)} prediction errors for classes 1 and 2")
    print(f"Error details saved to: {csv_path}")

if __name__ == '__main__':
    evaluate_model()
