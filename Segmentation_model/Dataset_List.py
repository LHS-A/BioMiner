class DatasetParameters:
    def __init__(self, dataset):
        self.dataset = dataset
        self.parameters = self.get_parameters()

    def get_parameters(self):
        dataset_params = {
            "CORN-1": {  
                "mode_metric": "nerve",
                "save_mode": "Nerve_for_paper",
                "lower_limit": 25, 
                "upper_limit": 165,
                "image_folder": "image",
                "label_folder": "label", 
                "nii": False,
                "roi_size": [384,384],
                "dialated_pixels_list": [1,2,3,4,5,6,7,8],
                "train_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3/train/label",
                "val_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3/val/label",
                "sen_thed": 0.83,
                "thed_lr_list": [0.70,0.81,0.82,0.83],
                "palette": [255], 
                "mapping":[255],
                "input_dim": 3,
                "num_classes": 1,
                "train_name" : "batch_1_2.png",
                "test_name" : "batch_2_14.png",
                "crop" : False,
                "resize" : False 
            },
            "CORN-Pro": { 
                "mode_metric": "nerve",
                "save_mode": "Nerve_for_paper",
                "lower_limit": 55, 
                "upper_limit": 235,
                "image_folder": "image",
                "label_folder": "label",  
                "roi_size": [384,384],
                "dialated_pixels_list": [6,5,4,3,2,1,0.1,0.001], 
                "nii": False, # whether 3D or not
                "train_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3_cell/train/label",
                "val_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3_cell/val/label",
                "sen_thed": 0.9,
                "thed_lr_list": [0.672,0.79,0.81,0.83],
                "palette": [255],
                "mapping":[255],
                "input_dim": 3,
                "num_classes": 1,
                "train_name" : "BHR(22).png",
                "test_name" : "HDQ(96).png",
                "crop" : False,
                "resize" : False
            }, 
            "CORN-Complex677": {
                "mode_metric": "nerve",
                "save_mode": "Nerve_for_paper",
                "lower_limit": 25,
                "upper_limit": 165,
                "image_folder": "image",
                "label_folder": "label",
                "nii": False,
                "roi_size": [384,384],
                "dialated_pixels_list": [1,2,3,4,5,6,7,8],
                "train_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3/train/label",
                "val_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3/val/label",
                "sen_thed": 0.83,
                "thed_lr_list": [0.70,0.81,0.82,0.83],
                "palette": [255], 
                "mapping":[255],
                "input_dim": 3, 
                "num_classes": 1,
                "train_name" : "BHR(10).png",
                "test_name" : "HDQ(96).png",
                "crop" : False,
                "resize" : False
            },  
            "Unseen-Data": {
                "mode_metric": "nerve",
                "save_mode": "Nerve_for_paper", 
                "lower_limit": 25,
                "upper_limit": 165,
                "image_folder": "image",
                "label_folder": "label",
                "nii": False,
                "roi_size": [384,384],
                "dialated_pixels_list": [1,2,3,4,5,6,7,8],
                "train_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3/train/label",
                "val_label_path": r"/home/ta/personal/LHS/DilNet_KD/Dataset/CORN_3/val/label",
                "sen_thed": 0.83,
                "thed_lr_list": [0.70,0.81,0.82,0.83],
                "palette": [255], 
                "mapping":[255],
                "input_dim": 3, 
                "num_classes": 1,
                "train_name" : "BHR(10).png",
                "test_name" : "HDQ(96).png",
                "crop" : False,
                "resize" : False
            }
        }
        return dataset_params.get(self.dataset, {})

