"""Documentation for this retained training component."""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5Model, T5ForConditionalGeneration
from tqdm import tqdm
import warnings
import sys
import argparse
from prettytable import PrettyTable
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    cohen_kappa_score, 
    roc_auc_score, 
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Compute the evaluation metrics.
sys.path.append(r"/data/Desktop/BioMiner")
from utils.evaluation_metrics import get_metrix

# Set the random seed for reproducibility.
def set_seed(seed: int = 42):
    """Documentation for this retained training component."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class NumpyEncoder(json.JSONEncoder):
    """Documentation for this retained training component."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure or use the encoder.
class T5EncoderForDualTasks(nn.Module):
    def __init__(self, base_model_path: str, feature_dim: int = 256, num_classes: int = 4,
                 use_pretrained_encoder: bool = False, pretrained_generative_model_path: str = None):
        """Documentation for this retained training component."""
        super().__init__() 
        
        print(f"加载基础T5模型: {base_model_path}")
        self.t5 = T5Model.from_pretrained(
            base_model_path,
            trust_remote_code=False
        )
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
        
        # Retain this implementation detail from the original training pipeline.
        self.classifier_nerve = nn.Linear(feature_dim, num_classes)  # Retain this implementation detail from the original training pipeline.
        self.classifier_cell = nn.Linear(feature_dim, num_classes)   # Retain this implementation detail from the original training pipeline.
        
        # Configure or use the encoder.
        for param in self.t5.decoder.parameters():
            param.requires_grad = False
        
        print(f"✓ 初始化T5双任务编码器")
        print(f"  编码器维度: {encoder_dim}")
        print(f"  特征维度: {feature_dim}")
        print(f"  任务1: 神经弯曲度分级")
        print(f"  任务2: 朗格汉斯细胞活化程度分级")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Configure or use the encoder.
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Retain this implementation detail from the original training pipeline.
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask
        
        # Extract or transform features.
        features = self.projection(pooled)
        
        # Retain this implementation detail from the original training pipeline.
        nerve_logits = self.classifier_nerve(features)  # Retain this implementation detail from the original training pipeline.
        cell_logits = self.classifier_cell(features)    # Retain this implementation detail from the original training pipeline.
        
        return nerve_logits, cell_logits

# Prepare or inspect the dataset.
class DualTaskJsonDataset(Dataset):
    """Documentation for this retained training component."""
    
    def __init__(self, json_path: str, tokenizer, max_len: int = 512):
        """Documentation for this retained training component."""
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.nerve_labels = []  # Process the task labels.
        self.cell_labels = []   # Process the task labels.
        self.texts = []
        self.image_names = []  # Retain this implementation detail from the original training pipeline.
        
        print(f"正在从JSON文件加载双任务数据: {json_path}")
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON文件应该包含一个列表")
            
            for i, item in enumerate(tqdm(data, desc="加载样本")):
                try:
                    # Validate the required condition.
                    if not isinstance(item, dict):
                        print(f"警告: 第{i}项不是字典格式，跳过")
                        continue
                    
                    # Validate the required condition.
                    if 'nerve_label' not in item or 'cell_label' not in item or 'input' not in item:
                        print(f"警告: 第{i}项缺少nerve_label、cell_label或input字段，跳过")
                        continue
                    
                    # Process the task labels.
                    nerve_label = item['nerve_label']
                    if not isinstance(nerve_label, int):
                        try:
                            nerve_label = int(nerve_label)
                        except:
                            print(f"警告: 第{i}项的nerve_label不是整数: {nerve_label}，跳过")
                            continue
                    
                    if nerve_label not in [0, 1, 2, 3]:
                        print(f"警告: 第{i}项的nerve_label {nerve_label} 不在有效范围 [0,1,2,3] 内，跳过")
                        continue
                    
                    # Process the task labels.
                    cell_label = item['cell_label']
                    if not isinstance(cell_label, int):
                        try:
                            cell_label = int(cell_label)
                        except:
                            print(f"警告: 第{i}项的cell_label不是整数: {cell_label}，跳过")
                            continue
                    
                    if cell_label not in [0, 1, 2, 3]:
                        print(f"警告: 第{i}项的cell_label {cell_label} 不在有效范围 [0,1,2,3] 内，跳过")
                        continue
                    
                    # Retrieve the required value.
                    text = item['input']
                    if not text or not isinstance(text, str):
                        print(f"警告: 第{i}项的文本为空或不是字符串，跳过")
                        continue
                    
                    # Retrieve the required value.
                    image_name = item.get('name', f"sample_{i}")
                    
                    self.samples.append(item)
                    self.nerve_labels.append(nerve_label)
                    self.cell_labels.append(cell_label)
                    self.texts.append(text)
                    self.image_names.append(image_name)
                    
                except Exception as e:
                    print(f"警告: 处理第{i}项时出错: {e}，跳过")
                    continue
            
            if len(self.samples) == 0:
                raise ValueError(f"在 {json_path} 中未找到任何有效的样本")
            
            print(f"✓ 加载了 {len(self.samples)} 个双任务样本")
            self._analyze_label_distribution()
            
        except Exception as e:
            raise ValueError(f"读取JSON文件失败: {e}")
    
    def _analyze_label_distribution(self):
        """Documentation for this retained training component."""
        nerve_labels_array = np.array(self.nerve_labels)
        cell_labels_array = np.array(self.cell_labels)
        num_samples = len(nerve_labels_array)
        
        print(f"\n双任务数据集统计信息:")
        print(f"总样本数: {num_samples}")
        
        # Process the task labels.
        self.nerve_levels = ["0级（无弯曲）", "1级（轻度弯曲）", "2级（中度弯曲）", "3级（重度弯曲）"]
        self.nerve_class_counts = np.bincount(nerve_labels_array, minlength=4)
        
        print(f"\n任务1 - 神经弯曲度分布:")
        for level in range(4):
            count = self.nerve_class_counts[level]
            ratio = count / num_samples
            print(f"  {self.nerve_levels[level]}: {count}个样本 ({ratio:.2%})")
        
        # Process the task labels.
        self.cell_levels = ["0级（无活化）", "1级（轻度活化）", "2级（中度活化）", "3级（重度活化）"]
        self.cell_class_counts = np.bincount(cell_labels_array, minlength=4)
        
        print(f"\n任务2 - 朗格汉斯细胞活化程度分布:")
        for level in range(4):
            count = self.cell_class_counts[level]
            ratio = count / num_samples
            print(f"  {self.cell_levels[level]}: {count}个样本 ({ratio:.2%})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        nerve_label = torch.tensor(self.nerve_labels[idx], dtype=torch.long)
        cell_label = torch.tensor(self.cell_labels[idx], dtype=torch.long)
        
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'nerve_label': nerve_label,  # Process the task labels.
            'cell_label': cell_label,    # Process the task labels.
            'image_name': self.image_names[idx]
        }

# Run the evaluation step.
class DualTaskEvaluator:
    def __init__(self, device: str = 'cuda'):
        """Documentation for this retained training component."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"✓ 双任务评估器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  任务1: 神经弯曲度分级")
        print(f"  任务2: 朗格汉斯细胞活化程度分级")
    
    def load_model(self, model_weights_path: str, base_model_path: str, 
                   tokenizer_path: str = None, feature_dim: int = 256, 
                   num_classes: int = 4, use_slow_tokenizer: bool = True):
        """Documentation for this retained training component."""
        print(f"\n正在加载双任务模型...")
        print(f"权重文件: {model_weights_path}")
        print(f"基础模型: {base_model_path}")
        print(f"分词器路径: {tokenizer_path if tokenizer_path else '与基础模型相同'}")
        
        # Configure or apply the tokenizer.
        print("\n1. 加载分词器...")
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=False,
                use_fast=not use_slow_tokenizer
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=False,
                use_fast=not use_slow_tokenizer
            )
        print(f"✓ Tokenizer加载完成")
        print(f"  词汇表大小: {self.tokenizer.vocab_size}")
        
        # Configure or use the model.
        print("\n2. 创建双任务模型结构...")
        self.model = T5EncoderForDualTasks(
            base_model_path=base_model_path,
            feature_dim=feature_dim,
            num_classes=num_classes,
            use_pretrained_encoder=False,
            pretrained_generative_model_path=None
        )
        
        # Load the required artifact.
        print(f"\n3. 加载双任务权重文件: {model_weights_path}")
        
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"权重文件不存在: {model_weights_path}")
        
        # Load the required artifact.
        state_dict = torch.load(model_weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        
        print(f"✓ 成功加载双任务权重文件")
        
        # Run the evaluation step.
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ 双任务模型已移至 {self.device}，设置为评估模式")
        
        # Configure or use the model.
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")

    # Compute the required value.
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_probs, num_classes=4):
        """Documentation for this retained training component."""
        metrics_per_class = []
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        total_samples = len(y_true)
        
        # Compute the required value.
        for i in range(num_classes):
            # Retain this implementation detail from the original training pipeline.
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = total_samples - (tp + fp + fn)
            
            # Retain this implementation detail from the original training pipeline.
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            acc_ovr = (tp + tn) / total_samples              # One-vs-Rest Accuracy
            f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
            
            # AUC (One-vs-Rest)
            try:
                # Process the task labels.
                y_true_binary = (y_true == i).astype(int)
                auc = roc_auc_score(y_true_binary, y_probs[:, i])
            except ValueError:
                auc = 0.0 # Retain this implementation detail from the original training pipeline.
                
            metrics_per_class.append({
                "acc": acc_ovr,
                "sens": sens,
                "spec": spec,
                "prec": prec,
                "f1": f1,
                "auc": auc,
                "support": tp + fn
            })

        # Compute the required value.
        supports = [m["support"] for m in metrics_per_class]
        total_support = sum(supports)
        
        weighted_avg = {}
        for key in ["acc", "sens", "spec", "prec", "f1", "auc"]:
            val = sum(m[key] * m["support"] for m in metrics_per_class) / total_support
            weighted_avg[key] = val
            
        # Compute the required value.
        kappa = cohen_kappa_score(y_true, y_pred)
        weighted_avg["kappa"] = kappa
        
        return metrics_per_class, weighted_avg
    
    def evaluate(self, dataloader: DataLoader, detailed: bool = True, 
                save_predictions: bool = True, predictions_file: str = None):
        """Documentation for this retained training component."""
        print("\n=== 评估双任务模型 ===")
        print("任务1: 神经弯曲度分级")
        print("任务2: 朗格汉斯细胞活化程度分级")
        
        # Retain this implementation detail from the original training pipeline.
        all_nerve_probs = []
        all_cell_probs = []
        all_nerve_logits = []
        all_cell_logits = []
        all_nerve_labels = []
        all_cell_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估双任务"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                nerve_labels = batch['nerve_label'].to(self.device)
                cell_labels = batch['cell_label'].to(self.device)
                image_names = batch['image_name']
                
                # Prepare or report the output.
                nerve_logits, cell_logits = self.model(input_ids, attention_mask)
                nerve_probs = torch.softmax(nerve_logits, dim=1)
                cell_probs = torch.softmax(cell_logits, dim=1)
                
                # Retain this implementation detail from the original training pipeline.
                all_nerve_probs.append(nerve_probs.cpu().numpy())
                all_cell_probs.append(cell_probs.cpu().numpy())
                all_nerve_logits.append(nerve_logits.cpu().numpy())
                all_cell_logits.append(cell_logits.cpu().numpy())
                all_nerve_labels.append(nerve_labels.cpu().numpy())
                all_cell_labels.append(cell_labels.cpu().numpy())
                all_image_names.extend(image_names)
        
        # Retain this implementation detail from the original training pipeline.
        all_nerve_probs = np.concatenate(all_nerve_probs, axis=0)
        all_cell_probs = np.concatenate(all_cell_probs, axis=0)
        all_nerve_logits = np.concatenate(all_nerve_logits, axis=0)
        all_cell_logits = np.concatenate(all_cell_logits, axis=0)
        all_nerve_labels = np.concatenate(all_nerve_labels, axis=0)
        all_cell_labels = np.concatenate(all_cell_labels, axis=0)
        
        # Collect or process predictions.
        nerve_preds = np.argmax(all_nerve_probs, axis=1)
        cell_preds = np.argmax(all_cell_probs, axis=1)
        
        # Compute the required value.
        n_metrics_cls, n_avg = self._calculate_comprehensive_metrics(all_nerve_labels, nerve_preds, all_nerve_probs)
        c_metrics_cls, c_avg = self._calculate_comprehensive_metrics(all_cell_labels, cell_preds, all_cell_probs)
        # -------------------------------------------------------------------------

        # Compute the required value.
        total_avg = {}
        for k in n_avg.keys():
            total_avg[k] = (n_avg[k] + c_avg[k]) / 2
        
        # Collect or process predictions.
        if save_predictions and predictions_file:
            self._save_dual_predictions(all_nerve_probs, all_cell_probs, all_nerve_labels, 
                                       all_cell_labels, all_image_names, nerve_preds, cell_preds, predictions_file)
        
        # Retain this implementation detail from the original training pipeline.
        if detailed:
            self._print_detailed_results(n_metrics_cls, n_avg, c_metrics_cls, c_avg, total_avg, len(all_nerve_labels))
        
        # Retain this implementation detail from the original training pipeline.
        results = {
            'mean_acc': total_avg['acc'],
            'mean_se': total_avg['sens'],
            'mean_sp': total_avg['spec'],
            'mean_auc': total_avg['auc'],
            'mean_f1': total_avg['f1'],
            'mean_kappa': total_avg['kappa'],
            'nerve_wAcc': [n_avg['acc']], # Retain this implementation detail from the original training pipeline.
            'nerve_wSe': [n_avg['sens']],
            'nerve_wSp': [n_avg['spec']],
            'cell_wAcc': [c_avg['acc']],
            'cell_wSe': [c_avg['sens']],
            'cell_wSp': [c_avg['spec']],
            'num_samples': len(all_nerve_labels)
        }
        
        return results, all_nerve_probs, all_cell_probs, all_nerve_labels, all_cell_labels, nerve_preds, cell_preds, all_image_names
    
    def _save_dual_predictions(self, all_nerve_probs, all_cell_probs, all_nerve_labels, 
                              all_cell_labels, all_image_names, nerve_preds, cell_preds, predictions_file):
        """Documentation for this retained training component."""
        print(f"\n正在保存双任务预测结果到: {predictions_file}")
        
        predictions_data = []
        for i in range(len(all_image_names)):
            # Collect or process predictions.
            prediction_entry = {
                "name": all_image_names[i],  # Retain this implementation detail from the original training pipeline.
                "text_nerve_prediction_probs": all_nerve_probs[i].tolist(),  # Collect or process predictions.
                "text_cell_prediction_probs": all_cell_probs[i].tolist(),    # Collect or process predictions.
                "true_nerve_label": int(all_nerve_labels[i]),  # Process the task labels.
                "true_cell_label": int(all_cell_labels[i])     # Process the task labels.
            }
            predictions_data.append(prediction_entry)
        
        # Ensure the output directory exists.
        predictions_dir = os.path.dirname(predictions_file)
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir, exist_ok=True)
        
        # Retain this implementation detail from the original training pipeline.
        if os.path.exists(predictions_file):
            os.remove(predictions_file)
            print(f"已删除旧的预测结果文件: {predictions_file}")
        
        # Save the generated artifact.
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"✓ 双任务预测结果已保存到: {predictions_file}")
        print(f"  保存了 {len(predictions_data)} 条预测记录")
        
        # Collect or process predictions.
        if len(predictions_data) > 0:
            print("\n预测结果示例（前3个样本）:")
            for i in range(min(3, len(predictions_data))):
                sample = predictions_data[i]
                print(f"  样本 {i+1}: {sample['name']}")
                print(f"    神经弯曲度预测概率: {[f'{p:.4f}' for p in sample['text_nerve_prediction_probs']]}")
                print(f"    朗格汉斯细胞预测概率: {[f'{p:.4f}' for p in sample['text_cell_prediction_probs']]}")
                print(f"    真实标签: 神经弯曲度={sample['true_nerve_label']}, 朗格汉斯细胞={sample['true_cell_label']}")
    
    def _print_detailed_results(self, n_metrics, n_avg, c_metrics, c_avg, total_avg, num_samples):
        """Documentation for this retained training component."""
        print("\n" + "=" * 90)
        print("双任务模型全指标详细评估表")
        print("=" * 90)
        
        print(f"样本总数: {num_samples}")
        
        # Create the required object.
        table = PrettyTable()
        # Retain this implementation detail from the original training pipeline.
        table.field_names = ["Task / Level", "Acc", "Sens", "Spec", "Prec", "F1", "Kappa", "AUC"]
        
        # Retain this implementation detail from the original training pipeline.
        for i, m in enumerate(n_metrics):
            table.add_row([
                f"Nerve - Level {i}", 
                f"{m['acc']:.4f}", f"{m['sens']:.4f}", f"{m['spec']:.4f}", 
                f"{m['prec']:.4f}", f"{m['f1']:.4f}", "-", f"{m['auc']:.4f}"
            ])
        # Retain this implementation detail from the original training pipeline.
        table.add_row([
            "Nerve - Average", 
            f"{n_avg['acc']:.4f}", f"{n_avg['sens']:.4f}", f"{n_avg['spec']:.4f}", 
            f"{n_avg['prec']:.4f}", f"{n_avg['f1']:.4f}", f"{n_avg['kappa']:.4f}", f"{n_avg['auc']:.4f}"
        ])
        
        table.add_row(["-"*15] + ["-"*6]*7) # Retain this implementation detail from the original training pipeline.
        
        # Retain this implementation detail from the original training pipeline.
        for i, m in enumerate(c_metrics):
            table.add_row([
                f"Cell - Level {i}", 
                f"{m['acc']:.4f}", f"{m['sens']:.4f}", f"{m['spec']:.4f}", 
                f"{m['prec']:.4f}", f"{m['f1']:.4f}", "-", f"{m['auc']:.4f}"
            ])
        # Retain this implementation detail from the original training pipeline.
        table.add_row([
            "Cell - Average", 
            f"{c_avg['acc']:.4f}", f"{c_avg['sens']:.4f}", f"{c_avg['spec']:.4f}", 
            f"{c_avg['prec']:.4f}", f"{c_avg['f1']:.4f}", f"{c_avg['kappa']:.4f}", f"{c_avg['auc']:.4f}"
        ])
        
        table.add_row(["="*15] + ["="*6]*7) # Retain this implementation detail from the original training pipeline.
        
        # Retain this implementation detail from the original training pipeline.
        table.add_row([
            "Total Average", 
            f"{total_avg['acc']:.4f}", f"{total_avg['sens']:.4f}", f"{total_avg['spec']:.4f}", 
            f"{total_avg['prec']:.4f}", f"{total_avg['f1']:.4f}", f"{total_avg['kappa']:.4f}", f"{total_avg['auc']:.4f}"
        ])
        
        print(table)
        print("注: 各Level指标采用One-vs-Rest计算; Average为加权平均; Kappa仅在任务级计算。")

# Run the evaluation step.
def evaluate_model(args):
    """Documentation for this retained training component."""
    # Set the random seed for reproducibility.
    set_seed(args.seed)
    
    print("=" * 80)
    print("双任务T5模型评估")
    print("任务1: 神经弯曲度分级")
    print("任务2: 朗格汉斯细胞活化程度分级")
    print("=" * 80)
    
    # Retain this implementation detail from the original training pipeline.
    if args.device == 'cuda':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print("警告: 请求使用CUDA但CUDA不可用，将使用CPU")
    elif args.device == 'cpu':
        device = "cpu"
    else:  # 'auto'
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    
    try:
        # Run the evaluation step.
        evaluator = DualTaskEvaluator(device=device)
        
        # Configure or use the model.
        evaluator.load_model(
            model_weights_path=args.model_weights_path,
            base_model_path=args.base_model,
            tokenizer_path=args.tokenizer_path,
            feature_dim=args.feature_dim,
            num_classes=4,
            use_slow_tokenizer=args.use_slow_tokenizer
        )
        
        # Prepare or inspect the dataset.
        print(f"\n加载双任务评估数据集: {args.eval_json}")
        dataset = DualTaskJsonDataset(
            json_path=args.eval_json,
            tokenizer=evaluator.tokenizer,
            max_len=args.max_len
        )
        
        # Create or use the data loader.
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0
        )
        print(f"✓ 创建数据加载器: {len(dataset)} 个样本, 批次大小: {args.batch_size}")
        
        # Run the evaluation step.
        results, nerve_probs, cell_probs, nerve_labels, cell_labels, nerve_preds, cell_preds, image_names = evaluator.evaluate(
            dataloader=dataloader,
            detailed=True,
            save_predictions=args.save_predictions,
            predictions_file=args.predictions_file
        )
        
        # Collect or process predictions.
        print(f"\n{'='*60}")
        print("导出类别1和2的预测错误")
        print(f"{'='*60}")
        
        # Collect or process predictions.
        nerve_error_data = []
        for i in range(len(nerve_labels)):
            true_label = nerve_labels[i]
            pred_label = nerve_preds[i]
            # Collect or process predictions.
            if (true_label == 0 or true_label == 3) and true_label != pred_label:
                nerve_error_data.append({
                    'image_name': image_names[i],
                    'true_label': int(true_label),
                    'prediction': int(pred_label),
                    'task': 'nerve_grading'
                })
        
        # Collect or process predictions.
        cell_error_data = []
        for i in range(len(cell_labels)):
            true_label = cell_labels[i]
            pred_label = cell_preds[i]
            # Collect or process predictions.
            if (true_label == 2) and true_label != pred_label:
                cell_error_data.append({
                    'image_name': image_names[i],
                    'true_label': int(true_label),
                    'prediction': int(pred_label),
                    'task': 'cell_activation'
                })
        
        # Retain this implementation detail from the original training pipeline.
        error_data = nerve_error_data + cell_error_data
        
        # Save the generated artifact.
        if error_data:
            error_df = pd.DataFrame(error_data)
            # Configure or use the model.
            csv_path = r"/data/Desktop/WSN/Dataset/Segmentation/Single_object/large/66878_Images/text_prediction_errors.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            error_df.to_csv(csv_path, sep='\t', index=False)
            
            print(f"任务1（神经弯曲度）发现 {len(nerve_error_data)} 个预测错误")
            print(f"任务2（朗格汉斯细胞）发现 {len(cell_error_data)} 个预测错误")
            print(f"总共发现 {len(error_data)} 个预测错误")
            print(f"错误详情保存到: {csv_path}")
            
            # Report the current status.
            # Collect or process predictions.
            # error_table = PrettyTable()
            # error_table.field_names = ["image_name", "task", "true_label", "prediction"]
            # for error in error_data[:10]:
            #     error_table.add_row([error['image_name'], error['task'], error['true_label'], error['prediction']])
            # if len(error_data) > 10:
            # Retain this implementation detail from the original training pipeline.
            # print(error_table)
        else:
            print(f"没有发现类别1和2的预测错误")
                  
        # Report the current status.
        print("\n" + "=" * 70)
        print("双任务评估完成总结")
        print("=" * 70)
        print(f"  模型权重: {args.model_weights_path}")
        print(f"  数据集: {args.eval_json}")
        print(f"  样本数: {len(nerve_labels)}")
        print(f"  平均准确率: {results['mean_acc']:.4f}")
        print(f"  任务1准确率（神经弯曲度）: {results['nerve_wAcc'][0]:.4f}")
        print(f"  任务2准确率（朗格汉斯细胞）: {results['cell_wAcc'][0]:.4f}")
        
        if args.save_predictions:
            print(f"  预测结果已保存: {args.predictions_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# Retain this implementation detail from the original training pipeline.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="双任务T5模型评估（神经弯曲度 + 朗格汉斯细胞活化程度）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Retain this implementation detail from the original training pipeline.
    parser.add_argument('--model_weights_path', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/checkpoint/Finetune_model/best_model_epoch_22_mean_acc_0.8034_nerve_acc_0.7445_cell_acc_0.8624.pth",
                        help='双任务模型.pth权重文件路径')
    
    parser.add_argument('--eval_json', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/datasets/CORN_LCs/Train_data/CCM_finetune_test.json",
                        help='双任务评估数据JSON文件路径')
    
    # Prepare or report the output.
    parser.add_argument('--predictions_file', type=str,
                        default="/data/Desktop/BioMiner/Generative_model/text_grading_predictions_CNs_LCs.json",
                        help='双任务预测结果保存路径')
    
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='是否保存预测结果')
    
    # Configure or use the model.
    parser.add_argument('--base_model', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/models/t5-clinical-base",
                        help='基础T5模型路径')
    
    parser.add_argument('--tokenizer_path', type=str, 
                        default="/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer_CNs_LCs/",
                        help='分词器路径')
    
    # Run the evaluation step.
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='特征维度')
    
    parser.add_argument('--max_len', type=int, default=512,
                        help='最大序列长度')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    parser.add_argument('--use_slow_tokenizer', action='store_true', default=False,
                        help='是否使用慢速分词器')
    
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='运行设备')
    
    args = parser.parse_args()
    
    # Resolve the required path.
    if not os.path.exists(args.model_weights_path):
        print(f"❌ 错误: 模型权重文件不存在: {args.model_weights_path}")
        exit(1)
    
    if not os.path.exists(args.eval_json):
        print(f"❌ 错误: 评估数据路径不存在: {args.eval_json}")
        exit(1)
    
    if not os.path.exists(args.base_model):
        print(f"❌ 错误: 基础模型路径不存在: {args.base_model}")
        exit(1)
    
    # Configure or apply the tokenizer.
    if args.tokenizer_path and not os.path.exists(args.tokenizer_path):
        print(f"⚠ 警告: 指定的分词器路径不存在: {args.tokenizer_path}")
        print("  将使用基础模型路径作为分词器路径")
        args.tokenizer_path = None
    
    print(f"\n配置检查:")
    print(f"  双任务模型权重: {args.model_weights_path}")
    print(f"  双任务评估数据: {args.eval_json}")
    print(f"  双任务预测结果保存路径: {args.predictions_file}")
    print(f"  保存预测结果: {args.save_predictions}")
    print(f"  基础模型: {args.base_model}")
    print(f"  分词器路径: {args.tokenizer_path if args.tokenizer_path else args.base_model}")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  最大长度: {args.max_len}")
    print(f"  随机种子: {args.seed}")
    print(f"  慢速分词器: {args.use_slow_tokenizer}")
    
    # Run the evaluation step.
    evaluate_model(args)