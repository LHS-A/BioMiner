# -- coding: utf-8 --
"""
@Model: BioMiner
@Time: 2025/11/11
@Author: lihongshuo
"""
import os
import torch
from Dataset_List import DatasetParameters

class Params():
    def __init__(self):
        # Remote server
        self.root_path = r"/home/lihongshuo/Desktop"
        self.model_name = "BioMiner"
        self.model_path = os.path.join(self.root_path, self.model_name)
# ==================================================== Values to be modified ================================================ 
        self.dataset = "CORN-Pro"  # CORN-Pro, CORN-1
        self.content = "UNet_600_test"
        self.switch_epoch = 200
        self.resume_training = False
        if self.resume_training == True:  # Must specify the route with pkl
            self.S_checkpoint_dir_path_last = r"/data/Desktop/BioMiner/Segmentation_model/checkpoint/BioMiner_2025MIA_CORN-Pro_600/best_checkpoint_199.pkl"

        self.env_name = "BioMiner_2025MIA_" + self.dataset + self.content
        self.data_path = r"/data/Desktop/BioMiner/Dataset/Segmentation_task/" + self.dataset

        dataset_params = DatasetParameters(self.dataset)
        self.mode_metric = dataset_params.parameters["mode_metric"]
        self.save_mode = dataset_params.parameters["save_mode"]
        self.lower_limit = dataset_params.parameters["lower_limit"]
        self.upper_limit = dataset_params.parameters["upper_limit"]
        self.image_folder = dataset_params.parameters["image_folder"]
        self.label_folder = dataset_params.parameters["label_folder"]
        self.roi_size = [dataset_params.parameters["roi_size"][0], dataset_params.parameters["roi_size"][1]]
        self.dialated_pixels_list = dataset_params.parameters["dialated_pixels_list"]
        self.nii = dataset_params.parameters["nii"]
        self.train_label_path = dataset_params.parameters["train_label_path"]
        self.val_label_path = dataset_params.parameters["val_label_path"]
        self.sen_thed = dataset_params.parameters["sen_thed"]
        self.thed_lr_list = dataset_params.parameters["thed_lr_list"]
        self.thed_lr = dataset_params.parameters["thed_lr_list"][0]
        self.palette = dataset_params.parameters["palette"]
        self.input_dim = dataset_params.parameters["input_dim"]
        self.num_classes = dataset_params.parameters["num_classes"]
        self.train_name = dataset_params.parameters["train_name"]
        self.test_name = dataset_params.parameters["test_name"]
        self.crop = dataset_params.parameters["crop"]
        self.resize = dataset_params.parameters["resize"]
        self.mapping = dataset_params.parameters["mapping"]
        dataset_params = DatasetParameters(self.dataset)
        # print(dataset_params.parameters)

        # ============================================================================================  
        self.device_ids = [0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ids = "0"
        self.save_pred = False

        self.open_OTUS = True
        self.vis_port = 9000
        self.epoch = 0
        self.init_lr = 1e-4     
        self.best_dice = 0 
        self.lr = 1e-4
        self.beta = 3  
        self.SDF = False  # If we use SDF as label.
        self.save_all_pkl = False
        self.epochs = 600
        self.enhance_mode = "train"  # "train"
        self.train_batch = 4
        self.val_batch = 4
        self.test_batch = 4  

        self.train_loss = []
        self.test_loss = [] 
        self.val_loss = []
    
        self.best_model_path = r"/data/Desktop/BioMiner/Segmentation_model/best_model/" + self.env_name + "/"
        os.makedirs(self.best_model_path, exist_ok=True)  
        self.checkpoint_dir_path = r"/data/Desktop/BioMiner/Segmentation_model/checkpoint/" + self.env_name + "/"
        os.makedirs(self.checkpoint_dir_path, exist_ok=True)      

        self.train_predict_data_path = r"/data/Desktop/BioMiner/Dataset/Segmentation_task/" + self.dataset + r"/train/" + "Predict_label" 
        self.val_predict_data_path = r"/data/Desktop/BioMiner/Dataset/Segmentation_task/" + self.dataset + r"/val/" + "Predict_label"
        self.test_predict_data_path = r"/data/Desktop/BioMiner/Dataset/Segmentation_task/" + self.dataset + r"/test/" + "Predict_label"
        os.makedirs(self.train_predict_data_path, exist_ok=True) 
        os.makedirs(self.val_predict_data_path, exist_ok=True) 
        os.makedirs(self.test_predict_data_path, exist_ok=True)
        self.paper_save_cell = r"/data/Desktop/BioMiner/Dataset/Segmentation_task/" + self.dataset + r"/test/" + "Final_pred"
        os.makedirs(self.paper_save_cell, exist_ok=True)

        self.metric_test = { 
        "total_sen_pred": [],  "total_dice_pred": [], "total_pre_pred": [], "total_fdr_pred":[], "total_mhd_pred": []}

        self.metrics_dict_test = {
                    "Sen_pred": {'total': self.metric_test["total_sen_pred"]},
                    "Dice_pred": {'total': self.metric_test["total_dice_pred"]},
                    "pre_pred": {'total': self.metric_test["total_pre_pred"]},
                    "FDR_pred": {'total': self.metric_test["total_fdr_pred"]},
                    "MHD_pred": {'total': self.metric_test["total_mhd_pred"]}
        }
