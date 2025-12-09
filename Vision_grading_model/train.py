"""
Classification Training Task Workflow:
1. Before execution, first run inference_or_label.py or inference_or_label_IMAGE.py in the segmentation task to obtain Predict_label.
2. Then train the grading task model. After obtaining the optimal weights, run utils/interpretability_CAM.py (for train, val, test) and cnn_visualization/gradcam.py (only for test, for qualitative visualization) to get the optimal ROI_image and paper interpretation figures.
3. Load the optimal checkpoint to fine-tune the grading model and obtain the optimal grading model.
Classification Label Inference Workflow:
1. First run the optimal segmentation model to get Predict_label.
2. Then load the optimal first-stage classification model to get the optimal ROI_image (see step 2 of the Classification Training Task Workflow).
3. Then call Vision_grading_model/inference_or_label.py to obtain the inference labels.
4. Filter the ideal labels corresponding to the disease.
"""
import sys
sys.path.append(r"/data/Desktop/BioMiner")   
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from Vision_grading_model.dataloader.grading_loader import MyData
import numpy as np
import datetime
from prettytable import PrettyTable
from losses.label_smooth import LabelSmoothingCrossEntropyLoss
from losses.wce_loss import WeightedCrossEntropyLoss
from Vision_grading_model.model.vision_grading import vision_grading_model
from utils.evaluation_metrics import get_metrix
import ssl  
import shutil
ssl._create_default_https_context = ssl._create_unverified_context

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

date_now = datetime.datetime.now().strftime("-%Y-%m-%d-")

args = {
    'root'      : '',    
    'dataset_name': 'CORN-LCs/1',  # CORN-LCs/1 BASELINE
    'epochs'    : 500,
    'lr'        : 0.0001,
    'test_step' : 1,
    'ckpt_path' : '/data/Desktop/BioMiner/Vision_grading_model/best_weight',
    'batch_size': 16,
    'pretrained': True, # backbone weights
    'checkpoint': True # checkpoint # For fine-tuning, simply set this to True, and no other code changes are needed; it will automatically load the current optimal weights.
}

base_path = f"/data/Desktop/BioMiner/Dataset/Grading_task/Activation/Grading_dataset/{args['dataset_name']}"
# base_path = f"/data/Desktop/BioMiner/Dataset/Grading_task/Activation/{args['dataset_name']}"
args['base_path'] = base_path  
args['ckpt_path'] = os.path.join(args['ckpt_path'], args['dataset_name']) 
os.makedirs(args['ckpt_path'], exist_ok=True)
args['train_sdf_data_path'] = f"{base_path}/train/SDF_image"
args['val_sdf_data_path'] = f"{base_path}/val/SDF_image" 
args['test_sdf_data_path'] = f"{base_path}/test/SDF_image"

args['train_predict_data_path'] = f"{base_path}/train/Predict_label"
args['val_predict_data_path'] = f"{base_path}/val/Predict_label"
args['test_predict_data_path'] = f"{base_path}/test/Predict_label"

args['train_ROI_data_path'] = f"{base_path}/train/ROI_image"
args['val_ROI_data_path'] = f"{base_path}/val/ROI_image"
args['test_ROI_data_path'] = f"{base_path}/test/ROI_image"

import os
for key in args:
    if 'path' in key and not key == 'root' and not key == 'base_path' and not key == 'ckpt_path':
        os.makedirs(args[key], exist_ok=True)

def load_latest_checkpoint(net):
    checkpoint_path = args['ckpt_path']
    
    if not os.path.exists(checkpoint_path):
        print("No checkpoint directory found, starting from scratch")
        return 0, 0
    
    model_files = [f for f in os.listdir(checkpoint_path) if f.startswith('best_model_epoch_') and f.endswith('.pth')]
    
    if not model_files:
        print("No checkpoint files found, starting from scratch")
        return 0, 0
    
    latest_model = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)), reverse=True)[0]
    latest_model_path = os.path.join(checkpoint_path, latest_model)
    
    print(f"Loading latest checkpoint: {latest_model_path}")
    
    try:
        checkpoint = torch.load(latest_model_path)
        net.load_state_dict(checkpoint)
        print("Model weights loaded successfully")
        
        filename = os.path.basename(latest_model_path)
        if 'epoch_' in filename and 'wacc_' in filename:
            try:
                epoch_part = filename.split('epoch_')[1].split('_')[0]
                wacc_part = filename.split('wacc_')[1].split('.pth')[0]
                start_epoch = int(epoch_part) + 1  # Start from the next epoch
                best_wacc = float(wacc_part)
                print(f"Resuming from epoch {start_epoch} with best wacc: {best_wacc:.4f}")
                return start_epoch, best_wacc
            except:
                print("Could not parse epoch and wacc from filename")
        
        print("Starting from epoch 0")
        return 0, 0
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting from scratch")
        return 0, 0
    
def create_required_folders():
    """
    Create SDF_image and ROI_image folders by copying from existing directories
    SDF_image: copy from image folder
    ROI_image: copy from Predict_label folder
    """
    print("Creating required folders: SDF_image and ROI_image...")
    
    for dataset_type in ['train', 'val', 'test']:
        # Define source and target paths
        image_source = os.path.join(base_path, dataset_type, 'image')
        predict_label_source = os.path.join(base_path, dataset_type, 'Predict_label')
        
        sdf_target = os.path.join(base_path, dataset_type, 'SDF_image')
        roi_target = os.path.join(base_path, dataset_type, 'ROI_image')
        
        # Create SDF_image folder by copying image folder
        if os.path.exists(sdf_target):
            # Check if folder is empty
            if not os.listdir(sdf_target):
                shutil.rmtree(sdf_target)
                print(f"Removed empty SDF_image folder: {sdf_target}")
                shutil.copytree(image_source, sdf_target)
                print(f"Created SDF_image folder by copying from image: {sdf_target}")
            else:
                print(f"SDF_image folder already exists with files: {sdf_target}")
        else:
            shutil.copytree(image_source, sdf_target)
            print(f"Created SDF_image folder by copying from image: {sdf_target}")
        
        # Create ROI_image folder by copying Predict_label folder
        if os.path.exists(roi_target):
            # Check if folder is empty
            if not os.listdir(roi_target):
                shutil.rmtree(roi_target)
                print(f"Removed empty ROI_image folder: {roi_target}")
                shutil.copytree(predict_label_source, roi_target)
                print(f"Created ROI_image folder by copying from Predict_label: {roi_target}")
            else:
                print(f"ROI_image folder already exists with files: {roi_target}")
        else:
            shutil.copytree(predict_label_source, roi_target)
            print(f"Created ROI_image folder by copying from Predict_label: {roi_target}")
    
    print("All required folders created successfully!")

def cleanup_old_models(max_keep=2):
    if not os.path.exists(args['ckpt_path']):
        return
    
    model_files = [f for f in os.listdir(args['ckpt_path']) if f.startswith('best_model_epoch_') and f.endswith('.pth')]
    if len(model_files) <= max_keep:
        return
    
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(args['ckpt_path'], x)), reverse=True)
    
    for old_model in model_files[max_keep:]:
        old_model_path = os.path.join(args['ckpt_path'], old_model)
        os.remove(old_model_path)
        print(f'--->Removed old model: {old_model} <---')

def save_best_model(net, epoch, wacc):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    model_path = os.path.join(args['ckpt_path'], f'best_model_epoch_{epoch}_wacc_{wacc:.4f}.pth')
    torch.save(net.state_dict(), model_path)
    print(f'--->Saved best model: {model_path} <---')
    
    cleanup_old_models(max_keep=2)

def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, os.path.join(args['ckpt_path'], 'DeepGrading-' + '.pth'))
    print('--->saved model:{}<--- '.format(args['root'] + args['ckpt_path']))


def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_acc(pred, label):
    pred = pred.data.cpu().numpy()
    label = label.data.cpu().numpy()
    right = pred == label
    right_count = len(right[right])
    return right_count / len(label)


def train():
    # Create required folders before training
    if not args['checkpoint']:
        create_required_folders()
    
    train_data = MyData(args['base_path'], train="train")
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=8, shuffle=True)

    net = vision_grading_model(num_class=4, model_name="resnet50", pre_train=args['pretrained']).cuda()
    
    if args['checkpoint']:
        start_epoch, best_wacc = load_latest_checkpoint(net)
    else:
        start_epoch = 0

    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    if args['pretrained']:
        ignored_params = list(map(id, net.fc1.parameters()))
        ignored_params += list(map(id, net.auxnet.parameters()))
        ignored_params += list(map(id, net.classifier.parameters()))
        
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        
        params_list = [{'params': base_params, 'lr': args['lr']}]
        params_list.append({'params': net.fc1.parameters(), 'lr': args['lr'] * 10})
        params_list.append({'params': net.auxnet.parameters(), 'lr': args['lr'] * 10})
        params_list.append({'params': net.classifier.parameters(), 'lr': args['lr'] * 10})
        
        optimizer = optim.SGD(params_list, lr=args['lr'], momentum=0.9, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(batchs_data), eta_min=1e-10)

    critrion_wce = WeightedCrossEntropyLoss().cuda()
    critrion_smooth = LabelSmoothingCrossEntropyLoss().cuda()
    print("---------------start training------------------")
    iters = 1
    best_wacc = 0
    best_metrics = None
    flag = 0
    for epoch in range(start_epoch, args['epochs']):
        net.train()
        epoch_num = 0
        epoch_loss = 0
        pred = []
        target = []
        for idx, batch in enumerate(batchs_data):
            img = batch[0]["img"].cuda()
            seg = batch[0]["seg"].cuda()
            roi = batch[0]["roi"].cuda()
            sdf = batch[0]["sdf"].cuda()
            image = torch.cat((img, img, seg), dim=1)
            roi = torch.cat((roi, roi), dim=1)
            class_id = batch[1]["img_id"].cuda()
            optimizer.zero_grad()
            x1, roi, predictions = net(image, roi)

            loss = critrion_smooth(predictions, class_id) + critrion_smooth(x1, class_id)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            class_id = class_id.data.cpu().numpy()
            pred.extend(predictions)
            target.extend(class_id)
            print("\t {0:d}---loss={1:8f}".format(iters, loss.item()))
            iters += 1
            epoch_num += 1
        print("Epoch {0:d}".format(epoch))
        pred = np.asarray(pred)
        target = np.asarray(target)
        wacc, wse, wsp = get_metrix(pred, target)
        scheduler.step()
        current_wacc, current_metrics = model_eval(net, epoch + 1)
        
        if current_metrics['mean_acc'] > best_wacc:
            flag = 0
            best_wacc = current_metrics['mean_acc']
            best_metrics = current_metrics
            save_best_model(net, epoch, best_wacc)
            
            print(f"Epoch {epoch} - New best model! Mean accuracy: {best_wacc:.4f}")
            print("Best model class metrics:")
            best_table = PrettyTable()
            best_table.field_names = ["Class", "Accuracy", "Sensitivity", "Specificity"]
            for i in range(len(best_metrics['class_acc'])):
                best_table.add_row([
                    f"Class {i+1}", 
                    f"{best_metrics['class_acc'][i]:.4f}", 
                    f"{best_metrics['class_se'][i]:.4f}", 
                    f"{best_metrics['class_sp'][i]:.4f}"
                ])
            print(best_table)
            print(f"Mean - Accuracy: {best_metrics['mean_acc']:.4f}, Sensitivity: {best_metrics['mean_se']:.4f}, Specificity: {best_metrics['mean_sp']:.4f}")
        else:
            flag += 1
            print(f"Epoch {epoch} - Current best mean accuracy: {best_wacc:.4f}")
            if best_metrics is not None:
                print("Historical best model class metrics:")
                history_table = PrettyTable()
                history_table.field_names = ["Class", "Accuracy", "Sensitivity", "Specificity"]
                for i in range(len(best_metrics['class_acc'])):
                    history_table.add_row([
                        f"Class {i+1}", 
                        f"{best_metrics['class_acc'][i]:.4f}", 
                        f"{best_metrics['class_se'][i]:.4f}", 
                        f"{best_metrics['class_sp'][i]:.4f}"
                    ])
                print(history_table)
                print(f"Mean - Accuracy: {best_metrics['mean_acc']:.4f}, Sensitivity: {best_metrics['mean_se']:.4f}, Specificity: {best_metrics['mean_sp']:.4f}")


def model_eval(net, iters):
    print("Start testing model...")
    test_data = MyData(args['base_path'], train="val")
    batchs_data = DataLoader(test_data, batch_size=args['batch_size'], num_workers=8)
    pred = []
    target = []
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
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            class_id = class_id.data.cpu().numpy()
            pred.extend(predictions)
            target.extend(class_id)
    pred = np.asarray(pred)
    target = np.asarray(target)
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
    return wacc[0], metrics


if __name__ == '__main__':
    train()
