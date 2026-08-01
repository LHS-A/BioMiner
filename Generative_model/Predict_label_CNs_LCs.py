import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5Model
from tqdm import tqdm
import sys
import argparse

# Set the random seed for reproducibility.
def set_seed(seed: int = 42):
    """Documentation for this retained evaluation component."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Configure or use the encoder.
class T5EncoderForDualTasks(nn.Module):
    def __init__(self, base_model_path: str, feature_dim: int = 256, num_classes: int = 4,
                 use_pretrained_encoder: bool = False, pretrained_generative_model_path: str = None):
        super().__init__() 
        
        print(f"加载基础T5模型: {base_model_path}")
        self.t5 = T5Model.from_pretrained(base_model_path, trust_remote_code=False)
        encoder_dim = self.t5.config.d_model
        
        # Extract or transform features.
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Retain this implementation detail from the original pipeline.
        self.classifier_nerve = nn.Linear(feature_dim, num_classes)
        self.classifier_cell = nn.Linear(feature_dim, num_classes)
        
        # Decode the generated sequences.
        for param in self.t5.decoder.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask
        
        features = self.projection(pooled)
        nerve_logits = self.classifier_nerve(features)
        cell_logits = self.classifier_cell(features)
        
        return nerve_logits, cell_logits

# Load or process the dataset.
class DualTaskJsonDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_len: int = 512):
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        print(f"正在加载数据: {json_path}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in tqdm(data, desc="解析样本"):
            if 'input' not in item:
                continue
            
            nerve_label = int(item.get('nerve_label', -1))
            cell_label = int(item.get('cell_label', -1))
            text = item['input']
            image_name = item.get('name', "unknown_image")
            
            self.samples.append({
                'text': text,
                'nerve_label': nerve_label,
                'cell_label': cell_label,
                'image_name': image_name
            })
            
        print(f"✓ 加载了 {len(self.samples)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image_name': item['image_name']
        }

# Generate or collect predictions.
class DualTaskPredictor:
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device_name = 'cuda'
            else:
                self.device_name = 'cpu'
        else:
            self.device_name = device
            
        self.device = torch.device(self.device_name)
        self.model = None
        self.tokenizer = None
        print(f"✓ 预测器已初始化，使用设备: {self.device}")
    
    def load_model(self, model_weights_path: str, base_model_path: str, 
                   tokenizer_path: str = None, feature_dim: int = 256, use_slow_tokenizer: bool = True):
        
        print(f"\n加载模型权重: {model_weights_path}")
        
        tk_path = tokenizer_path if tokenizer_path else base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tk_path, trust_remote_code=False, use_fast=not use_slow_tokenizer)
        
        self.model = T5EncoderForDualTasks(
            base_model_path=base_model_path,
            feature_dim=feature_dim,
            num_classes=4
        )
        
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"模型权重文件不存在: {model_weights_path}")
            
        state_dict = torch.load(model_weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("✓ 模型加载完成")

    def predict_and_save(self, dataloader: DataLoader, output_file: str, json_file_to_update: str):
        """Documentation for this retained evaluation component."""
        print(f"\n开始推理...")
        print(f"1. 预测结果TXT将保存至: {output_file}")
        print(f"2. 同步更新JSON文件: {json_file_to_update}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Generate or collect predictions.
        predictions_map = {}
        txt_results = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="正在推理"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image_names = batch['image_name']
                
                # Retain this implementation detail from the original pipeline.
                nerve_logits, cell_logits = self.model(input_ids, attention_mask)
                
                # Generate or collect predictions.
                nerve_preds = torch.argmax(nerve_logits, dim=1).cpu().numpy()
                cell_preds = torch.argmax(cell_logits, dim=1).cpu().numpy()
                
                # Retain this implementation detail from the original pipeline.
                for name, n_pred, c_pred in zip(image_names, nerve_preds, cell_preds):
                    # Write the output data.
                    txt_results.append(f"{name}\t{n_pred}\t{c_pred}")
                    
                    # Update the current state.
                    predictions_map[name] = {
                        'nerve_label': int(n_pred),
                        'cell_label': int(c_pred)
                    }
        
        # Save the generated artifact.
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in txt_results:
                f.write(line + "\n")
        print(f"✓ TXT文件已生成，共 {len(txt_results)} 条记录。")

        # Update the current state.
        self._update_json_labels(json_file_to_update, predictions_map)

    def _update_json_labels(self, json_path: str, predictions_map: Dict):
        """Documentation for this retained evaluation component."""
        print(f"\n正在更新JSON文件标签...")
        
        if not os.path.exists(json_path):
            print(f"❌ 错误: 要更新的JSON文件不存在: {json_path}")
            return

        try:
            # Load or process the dataset.
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            update_count = 0
            
            # Update the current state.
            for item in data:
                img_name = item.get('name')
                if img_name in predictions_map:
                    # Generate or collect predictions.
                    pred = predictions_map[img_name]
                    
                    # Process the grading labels.
                    item['nerve_label'] = pred['nerve_label']
                    item['cell_label'] = pred['cell_label']
                    
                    update_count += 1
            
            # Retain this implementation detail from the original pipeline.
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"✓ JSON文件更新完成！已修正 {update_count} 个样本的标签。")
            print(f"  文件路径: {json_path}")
            
        except Exception as e:
            print(f"❌ 更新JSON文件时出错: {e}")

# Retain this implementation detail from the original pipeline.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双任务模型推理预测与标签修正")
    
    # Resolve the required path.
    parser.add_argument('--model_weights_path', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/checkpoint/Finetune_model/best_model_epoch.pth",
                        help='模型权重路径')
    
    parser.add_argument('--eval_json', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/datasets/Both_corpus/Train_data/CCM_finetune_test.json",
                        help='测试集JSON路径 (用于读取数据输入，同时也会被更新标签)')
    
    parser.add_argument('--output_file', type=str,
                        default="/data/Desktop/WSN/Dataset/Segmentation/Single_object/large/66878_Images/test/test.txt",
                        help='预测结果保存的TXT文件路径')
    
    # Configure or use the model.
    parser.add_argument('--base_model', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/models/t5-clinical-base")
    parser.add_argument('--tokenizer_path', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer_CNs_LCs/")
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Generate or collect predictions.
    predictor = DualTaskPredictor(device=args.device)
    
    # Configure or use the model.
    predictor.load_model(
        model_weights_path=args.model_weights_path,
        base_model_path=args.base_model,
        tokenizer_path=args.tokenizer_path,
        feature_dim=args.feature_dim
    )
    
    # Load or process the dataset.
    dataset = DualTaskJsonDataset(
        json_path=args.eval_json,
        tokenizer=predictor.tokenizer,
        max_len=args.max_len
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Generate or collect predictions.
    # Process the grading labels.
    predictor.predict_and_save(dataloader, args.output_file, args.eval_json)