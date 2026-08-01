"""Documentation for this retained training component."""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import warnings
import sys
import argparse
from prettytable import PrettyTable
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


# Prepare or inspect the dataset.
class DualTaskJsonDataset(Dataset):
    """Documentation for this retained training component."""
    
    def __init__(self, json_path: str, tokenizer: T5Tokenizer, max_len: int = 512):
        """Documentation for this retained training component."""
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.nerve_labels = []  # Process the task labels.
        self.cell_labels = []   # Process the task labels.
        self.texts = []
        self.image_names = []  # Retain this implementation detail from the original training pipeline.
        
        print(f"正在从JSON文件加载数据: {json_path}")
        
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
                    required_fields = ['nerve_label', 'cell_label', 'input']
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        print(f"警告: 第{i}项缺少字段 {missing_fields}，跳过")
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
            
            print(f"✓ 加载了 {len(self.samples)} 个样本")
            self._analyze_label_distribution()
            
        except Exception as e:
            raise ValueError(f"读取JSON文件失败: {e}")
    
    def _analyze_label_distribution(self):
        """Documentation for this retained training component."""
        nerve_labels_array = np.array(self.nerve_labels)
        cell_labels_array = np.array(self.cell_labels)
        num_samples = len(nerve_labels_array)
        
        print(f"\n数据集统计信息:")
        print(f"总样本数: {num_samples}")
        
        # Process the task labels.
        nerve_levels = ["0级（无弯曲）", "1级（轻度弯曲）", "2级（中度弯曲）", "3级（重度弯曲）"]
        print(f"\n任务1 - 神经弯曲度分布:")
        
        nerve_class_counts = np.bincount(nerve_labels_array, minlength=4)
        for level in range(4):
            count = nerve_class_counts[level]
            ratio = count / num_samples
            print(f"  {nerve_levels[level]}: {count}个样本 ({ratio:.2%})")
        
        # Process the task labels.
        cell_levels = ["0级（无活化）", "1级（轻度活化）", "2级（中度活化）", "3级（重度活化）"]
        print(f"\n任务2 - 朗格汉斯细胞活化程度分布:")
        
        cell_class_counts = np.bincount(cell_labels_array, minlength=4)
        for level in range(4):
            count = cell_class_counts[level]
            ratio = count / num_samples
            print(f"  {cell_levels[level]}: {count}个样本 ({ratio:.2%})")
        
        # Save the generated artifact.
        self.nerve_class_counts = nerve_class_counts
        self.cell_class_counts = cell_class_counts
        self.nerve_class_names = nerve_levels
        self.cell_class_names = cell_levels
    
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


# Configure or use the encoder.
class T5EncoderForDualTasks(nn.Module):
    def __init__(self, base_model_path: str, tokenizer_path: str = None, feature_dim: int = 256, 
                 num_classes: int = 4, use_pretrained_encoder: bool = True, 
                 pretrained_generative_model_path: str = None, use_slow_tokenizer: bool = True):
        """Documentation for this retained training component."""
        super().__init__()
        
        # Configure or use the model.
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
        
        print(f"✓ 初始化T5编码器 + 双任务四分类头")
        print(f"  编码器维度: {encoder_dim}")
        print(f"  特征维度: {feature_dim}")
        print(f"  任务1: 角膜神经弯曲度分级")
        print(f"  任务2: 朗格汉斯细胞活化程度分级")
        
        # Run the training step.
        if use_pretrained_encoder and pretrained_generative_model_path:
            self._load_generative_model_weights(pretrained_generative_model_path)
    
    def _load_generative_model_weights(self, generative_model_path: str):
        """Documentation for this retained training component."""
        print(f"\n正在从生成模型加载编码器权重: {generative_model_path}")
        
        if not os.path.exists(generative_model_path):
            print(f"警告: 生成模型路径不存在: {generative_model_path}")
            return
        
        try:
            # Configure or use the model.
            generative_model = T5ForConditionalGeneration.from_pretrained(
                generative_model_path,
                trust_remote_code=False
            )
            
            # Configure or use the encoder.
            generative_encoder_state_dict = generative_model.encoder.state_dict()
            
            # Configure or use the encoder.
            current_encoder_state_dict = self.t5.encoder.state_dict()
            
            # Compute the required value.
            matched_layers = 0
            total_layers = len(current_encoder_state_dict)
            
            # Load the required artifact.
            for key in current_encoder_state_dict:
                if key in generative_encoder_state_dict:
                    # Validate the required condition.
                    if current_encoder_state_dict[key].shape == generative_encoder_state_dict[key].shape:
                        current_encoder_state_dict[key] = generative_encoder_state_dict[key].clone()
                        matched_layers += 1
                    else:
                        print(f"警告: 层 {key} 形状不匹配，跳过")
                        print(f"  当前形状: {current_encoder_state_dict[key].shape}")
                        print(f"  生成模型形状: {generative_encoder_state_dict[key].shape}")
            
            # Configure or use the encoder.
            self.t5.encoder.load_state_dict(current_encoder_state_dict)
            
            print(f"✓ 成功加载编码器权重: {matched_layers}/{total_layers} 层匹配")
            
        except Exception as e:
            print(f"❌ 加载生成模型权重失败: {e}")
    
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


# Retain this implementation detail from the original training pipeline.
class DualTaskClassifier:
    def __init__(self, base_model_path: str, tokenizer_path: str = None, feature_dim: int = 256, 
                 num_classes: int = 4, device: str = 'cuda', use_pretrained_encoder: bool = False,
                 pretrained_generative_model_path: str = None, use_slow_tokenizer: bool = True):
        """Documentation for this retained training component."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.base_model_path = base_model_path
        self.tokenizer_path = tokenizer_path
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_slow_tokenizer = use_slow_tokenizer
        
        print(f"✓ 双任务四分类器初始化")
        print(f"  设备: {self.device}")
        print(f"  基础模型路径: {base_model_path}")
        print(f"  分词器路径: {tokenizer_path if tokenizer_path else '与模型路径相同'}")
        print(f"  特征维度: {feature_dim}")
        print(f"  每个任务类别数: {num_classes}")
        print(f"  使用慢速分词器: {use_slow_tokenizer}")
        print(f"  任务1: 角膜神经弯曲度分级")
        print(f"  任务2: 朗格汉斯细胞活化程度分级")
        
        # Configure or use the model.
        
    def _init_model(self, use_pretrained_encoder: bool, pretrained_generative_model_path: str):
        """Documentation for this retained training component."""
        self.model = T5EncoderForDualTasks(
            base_model_path=self.base_model_path,
            tokenizer_path=self.tokenizer_path,
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            use_pretrained_encoder=use_pretrained_encoder,
            pretrained_generative_model_path=pretrained_generative_model_path,
            use_slow_tokenizer=self.use_slow_tokenizer
        )
        
        self.use_pretrained_encoder = use_pretrained_encoder
        self.pretrained_generative_model_path = pretrained_generative_model_path
        
        if use_pretrained_encoder and pretrained_generative_model_path:
            print(f"  已加载生成模型编码器权重: {pretrained_generative_model_path}")
    
    def _save_predictions(self, dataloader: DataLoader, epoch: int, mean_acc: float, nerve_acc: float, cell_acc: float):
        """Documentation for this retained training component."""
        predictions_file = "/data/Desktop/BioMiner/Generative_model/text_grading_predictions_CNs_LCs.json"
        print(f"\n正在保存最优模型的预测结果到: {predictions_file}")
        
        self.model.eval()
        all_nerve_probs = []
        all_cell_probs = []
        all_nerve_labels = []
        all_cell_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="生成预测结果"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                nerve_labels = batch['nerve_label'].to(self.device)
                cell_labels = batch['cell_label'].to(self.device)
                image_names = batch['image_name']
                
                nerve_logits, cell_logits = self.model(input_ids, attention_mask)
                nerve_probs = torch.softmax(nerve_logits, dim=1)
                cell_probs = torch.softmax(cell_logits, dim=1)
                
                all_nerve_probs.append(nerve_probs.cpu().numpy())
                all_cell_probs.append(cell_probs.cpu().numpy())
                all_nerve_labels.append(nerve_labels.cpu().numpy())
                all_cell_labels.append(cell_labels.cpu().numpy())
                all_image_names.extend(image_names)
        
        all_nerve_probs = np.concatenate(all_nerve_probs, axis=0)
        all_cell_probs = np.concatenate(all_cell_probs, axis=0)
        all_nerve_labels = np.concatenate(all_nerve_labels, axis=0)
        all_cell_labels = np.concatenate(all_cell_labels, axis=0)
        
        # Collect or process predictions.
        nerve_preds = np.argmax(all_nerve_probs, axis=1)
        cell_preds = np.argmax(all_cell_probs, axis=1)
        
        # Collect or process predictions.
        predictions_data = []
        for i in range(len(all_image_names)):
            predictions_data.append({
                'name': all_image_names[i],
                'text_nerve_prediction_probs': all_nerve_probs[i].tolist(),
                'text_cell_prediction_probs': all_cell_probs[i].tolist(),
                'true_nerve_label': int(all_nerve_labels[i]),
                'true_cell_label': int(all_cell_labels[i]),
                'epoch': epoch,
                'mean_acc': mean_acc,
                'nerve_acc': nerve_acc,
                'cell_acc': cell_acc
            })
        
        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"✓ 双任务预测结果已保存到: {predictions_file}")
        print(f"  保存了 {len(predictions_data)} 条预测记录（来自第 {epoch} 轮，平均准确率: {mean_acc:.4f}）")

    def _cleanup_old_models(self, keep_num: int = 4):
        """Documentation for this retained training component."""
        if not hasattr(self, 'output_dir') or not self.output_dir:
            return
        
        try:
            # Ensure the output directory exists.
            model_files = []
            for file in os.listdir(self.output_dir):
                if file.endswith('.pth'):
                    file_path = os.path.join(self.output_dir, file)
                    if os.path.isfile(file_path):
                        # Retrieve the required value.
                        mtime = os.path.getmtime(file_path)
                        model_files.append((file_path, mtime))
            
            # Retain this implementation detail from the original training pipeline.
            if len(model_files) > keep_num:
                # Retain this implementation detail from the original training pipeline.
                model_files.sort(key=lambda x: x[1], reverse=True)
                
                # Retain this implementation detail from the original training pipeline.
                files_to_keep = [f[0] for f in model_files[:keep_num]]
                
                # Retain this implementation detail from the original training pipeline.
                for file_path, _ in model_files[keep_num:]:
                    try:
                        os.remove(file_path)
                        print(f"✓ 删除旧模型文件: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"⚠ 删除文件 {file_path} 失败: {e}")
                
                print(f"✓ 已清理旧模型文件，保留最新的 {keep_num} 个文件")
        
        except Exception as e:
            print(f"⚠ 清理模型文件时出错: {e}")

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None,
            num_epochs: int = 30, lr: float = 1e-4, weight_method='inverse',
            freeze_encoder: bool = False, use_pretrained_encoder: bool = False,
            pretrained_generative_model_path: str = None, output_dir: str = None):
        """Documentation for this retained training component."""
        # Configure or use the model.
        self._init_model(use_pretrained_encoder, pretrained_generative_model_path)
        self.model.to(self.device)
        
        # Ensure the output directory exists.
        self.output_dir = output_dir
        
        # Configure or use the encoder.
        if freeze_encoder:
            for param in self.model.t5.encoder.parameters():
                param.requires_grad = False
            print("✓ 冻结编码器参数，只训练分类头和投影层")
        
        # Compute the required value.
        print("\n计算双任务类别权重...")
        
        # Retain this implementation detail from the original training pipeline.
        train_nerve_labels = []
        train_cell_labels = []
        for batch in train_dataloader:
            train_nerve_labels.append(batch['nerve_label'].numpy())
            train_cell_labels.append(batch['cell_label'].numpy())
        
        if train_nerve_labels and train_cell_labels:
            train_nerve_labels_array = np.concatenate(train_nerve_labels, axis=0)
            train_cell_labels_array = np.concatenate(train_cell_labels, axis=0)
            
            # Retain this implementation detail from the original training pipeline.
            nerve_class_counts = np.bincount(train_nerve_labels_array, minlength=self.num_classes)
            nerve_total_samples = len(train_nerve_labels_array)
            nerve_class_weights = nerve_total_samples / (self.num_classes * nerve_class_counts + 1e-9)
            nerve_class_weights = torch.tensor(nerve_class_weights, dtype=torch.float32).to(self.device)
            
            # Retain this implementation detail from the original training pipeline.
            cell_class_counts = np.bincount(train_cell_labels_array, minlength=self.num_classes)
            cell_total_samples = len(train_cell_labels_array)
            cell_class_weights = cell_total_samples / (self.num_classes * cell_class_counts + 1e-9)
            cell_class_weights = torch.tensor(cell_class_weights, dtype=torch.float32).to(self.device)
            
            print(f"任务1（神经弯曲度）类别分布: {nerve_class_counts.tolist()}")
            print(f"任务1（神经弯曲度）类别权重: {nerve_class_weights.cpu().numpy()}")
            print(f"任务2（朗格汉斯细胞）类别分布: {cell_class_counts.tolist()}")
            print(f"任务2（朗格汉斯细胞）类别权重: {cell_class_weights.cpu().numpy()}")
        else:
            nerve_class_weights = None
            cell_class_weights = None
        
        # Run the training step.
        if freeze_encoder:
            # Run the training step.
            trainable_params = list(self.model.projection.parameters()) + \
                              list(self.model.classifier_nerve.parameters()) + \
                              list(self.model.classifier_cell.parameters())
            print(f"训练参数: 投影层 + 双任务分类头")
        else:
            # Run the training step.
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            print(f"训练参数: 编码器 + 投影层 + 双任务分类头")
        
        # Configure the optimizer.
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr, weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # Run the training step.
        history = {
            'train_loss': [], 'val_loss': [],
            'val_mean_acc': [], 'val_nerve_acc': [], 'val_cell_acc': []
        }
        
        best_val_mean_acc = -1.0  # Retain this implementation detail from the original training pipeline.
        best_epoch = 0
        patience_counter = 0
        patience = 10
        
        print(f"\n=== 开始双任务训练 ({num_epochs} epochs) ===")
        
        for epoch in range(num_epochs):
            # Run the training step.
            self.model.train()
            total_loss = 0.0
            train_steps = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                nerve_labels = batch['nerve_label'].to(self.device)
                cell_labels = batch['cell_label'].to(self.device)
                
                optimizer.zero_grad()
                nerve_logits, cell_logits = self.model(input_ids, attention_mask)
                
                # Compute the training loss.
                if nerve_class_weights is not None and cell_class_weights is not None:
                    loss_fn_nerve = nn.CrossEntropyLoss(weight=nerve_class_weights)
                    loss_fn_cell = nn.CrossEntropyLoss(weight=cell_class_weights)
                else:
                    loss_fn_nerve = nn.CrossEntropyLoss()
                    loss_fn_cell = nn.CrossEntropyLoss()
                
                loss_nerve = loss_fn_nerve(nerve_logits, nerve_labels)
                loss_cell = loss_fn_cell(cell_logits, cell_labels)
                
                # Compute the training loss.
                total_batch_loss = loss_nerve + loss_cell
                total_batch_loss.backward()
                
                # Retain this implementation detail from the original training pipeline.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                train_steps += 1
            
            avg_train_loss = total_loss / train_steps
            history['train_loss'].append(avg_train_loss)
            
            # Run the validation step.
            if val_dataloader:
                val_loss, val_mean_acc, val_nerve_acc, val_cell_acc, val_metrics = self._evaluate_epoch(val_dataloader, nerve_class_weights, cell_class_weights)
                
                history['val_loss'].append(val_loss)
                history['val_mean_acc'].append(val_mean_acc)
                history['val_nerve_acc'].append(val_nerve_acc)
                history['val_cell_acc'].append(val_cell_acc)
                
                print(f"\nEpoch {epoch + 1}:")
                print(f"  训练损失: {avg_train_loss:.4f}")
                print(f"  验证损失: {val_loss:.4f}")
                
                # Run the validation step.
                table = PrettyTable()
                table.field_names = ["任务", "准确率", "敏感度", "特异度"]
                table.add_row(["神经弯曲度", f"{val_metrics['nerve_acc']:.4f}", f"{val_metrics['nerve_se']:.4f}", f"{val_metrics['nerve_sp']:.4f}"])
                table.add_row(["朗格汉斯细胞", f"{val_metrics['cell_acc']:.4f}", f"{val_metrics['cell_se']:.4f}", f"{val_metrics['cell_sp']:.4f}"])
                table.add_row(["平均", f"{val_mean_acc:.4f}", f"{val_metrics['mean_se']:.4f}", f"{val_metrics['mean_sp']:.4f}"])
                print(table)
                
                # Retain this implementation detail from the original training pipeline.
                if val_mean_acc > best_val_mean_acc:
                    best_val_mean_acc = val_mean_acc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    
                    # Handle the training checkpoint.
                    checkpoint_path = f"{self.output_dir}/best_model_epoch_{best_epoch}_mean_acc_{best_val_mean_acc:.4f}_nerve_acc_{val_nerve_acc:.4f}_cell_acc_{val_cell_acc:.4f}.pth"
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"★ 保存最佳模型: {checkpoint_path}")
                    
                    # Configure or use the model.
                    self._cleanup_old_models()

                    # Collect or process predictions.
                    self._save_predictions(val_dataloader, best_epoch, best_val_mean_acc, val_nerve_acc, val_cell_acc)
                    
                    # Configure or use the model.
                    print("最佳模型各类别指标:")
                    
                    # Retain this implementation detail from the original training pipeline.
                    nerve_table = PrettyTable()
                    nerve_table.field_names = ["神经弯曲度", "准确率", "敏感度", "特异度"]
                    for i in range(len(val_metrics['nerve_class_acc'])):
                        nerve_table.add_row([
                            f"等级 {i+1}", 
                            f"{val_metrics['nerve_class_acc'][i]:.4f}", 
                            f"{val_metrics['nerve_class_se'][i]:.4f}", 
                            f"{val_metrics['nerve_class_sp'][i]:.4f}"
                        ])
                    print("任务1（神经弯曲度）:")
                    print(nerve_table)
                    
                    # Retain this implementation detail from the original training pipeline.
                    cell_table = PrettyTable()
                    cell_table.field_names = ["朗格汉斯细胞", "准确率", "敏感度", "特异度"]
                    for i in range(len(val_metrics['cell_class_acc'])):
                        cell_table.add_row([
                            f"等级 {i+1}", 
                            f"{val_metrics['cell_class_acc'][i]:.4f}", 
                            f"{val_metrics['cell_class_se'][i]:.4f}", 
                            f"{val_metrics['cell_class_sp'][i]:.4f}"
                        ])
                    print("任务2（朗格汉斯细胞）:")
                    print(cell_table)
                    
                    print(f"平均指标 - 准确率: {val_metrics['mean_acc']:.4f}, 敏感度: {val_metrics['mean_se']:.4f}, 特异度: {val_metrics['mean_sp']:.4f}")
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch + 1} - 当前最佳平均准确率: {best_val_mean_acc:.4f} (epoch {best_epoch})")
                    if patience_counter >= patience:
                        print(f"早停触发，最佳平均准确率: {best_val_mean_acc:.4f}")
                        break
            else:
                print(f"Epoch {epoch + 1}: 训练损失: {avg_train_loss:.4f}")
            
            scheduler.step()
        
        # Configure or use the model.
        best_model_path = f"{self.output_dir}/best_model_epoch_{best_epoch}_mean_acc_{best_val_mean_acc:.4f}_nerve_acc_*_cell_acc_*.pth"
        model_files = [f for f in os.listdir(self.output_dir) if f.startswith(f'best_model_epoch_{best_epoch}_')]
        
        if model_files:
            best_model_path = os.path.join(self.output_dir, model_files[0])
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"★ 加载最佳模型: {best_model_path}")
        
        return history
    
    def _evaluate_epoch(self, dataloader: DataLoader, nerve_class_weights=None, cell_class_weights=None):
        """Documentation for this retained training component."""
        self.model.eval()
        total_loss = 0.0
        all_nerve_preds = []
        all_cell_preds = []
        all_nerve_labels = []
        all_cell_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                nerve_labels = batch['nerve_label'].to(self.device)
                cell_labels = batch['cell_label'].to(self.device)
                
                nerve_logits, cell_logits = self.model(input_ids, attention_mask)
                
                # Compute the training loss.
                if nerve_class_weights is not None and cell_class_weights is not None:
                    loss_fn_nerve = nn.CrossEntropyLoss(weight=nerve_class_weights)
                    loss_fn_cell = nn.CrossEntropyLoss(weight=cell_class_weights)
                else:
                    loss_fn_nerve = nn.CrossEntropyLoss()
                    loss_fn_cell = nn.CrossEntropyLoss()
                
                loss_nerve = loss_fn_nerve(nerve_logits, nerve_labels)
                loss_cell = loss_fn_cell(cell_logits, cell_labels)
                total_batch_loss = loss_nerve + loss_cell
                total_loss += total_batch_loss.item()
                
                # Collect or process predictions.
                nerve_preds = torch.argmax(nerve_logits, dim=1)
                cell_preds = torch.argmax(cell_logits, dim=1)
                
                all_nerve_preds.append(nerve_preds.cpu().numpy())
                all_cell_preds.append(cell_preds.cpu().numpy())
                all_nerve_labels.append(nerve_labels.cpu().numpy())
                all_cell_labels.append(cell_labels.cpu().numpy())
        
        all_nerve_preds = np.concatenate(all_nerve_preds, axis=0)
        all_cell_preds = np.concatenate(all_cell_preds, axis=0)
        all_nerve_labels = np.concatenate(all_nerve_labels, axis=0)
        all_cell_labels = np.concatenate(all_cell_labels, axis=0)
        
        # Compute the required value.
        nerve_wAcc, nerve_wSe, nerve_wSp = get_metrix(all_nerve_preds, all_nerve_labels)
        cell_wAcc, cell_wSe, cell_wSp = get_metrix(all_cell_preds, all_cell_labels)
        
        # Compute the required value.
        mean_acc = (nerve_wAcc[0] + cell_wAcc[0]) / 2
        mean_se = (nerve_wSe[0] + cell_wSe[0]) / 2
        mean_sp = (nerve_wSp[0] + cell_wSp[0]) / 2
        
        avg_loss = total_loss / len(dataloader)
        
        # Retain this implementation detail from the original training pipeline.
        metrics = {
            'mean_acc': mean_acc,
            'mean_se': mean_se,
            'mean_sp': mean_sp,
            'nerve_acc': nerve_wAcc[0],
            'nerve_se': nerve_wSe[0],
            'nerve_sp': nerve_wSp[0],
            'cell_acc': cell_wAcc[0],
            'cell_se': cell_wSe[0],
            'cell_sp': cell_wSp[0],
            'nerve_class_acc': nerve_wAcc[1] if len(nerve_wAcc) > 1 else [nerve_wAcc[0]],
            'nerve_class_se': nerve_wSe[1] if len(nerve_wSe) > 1 else [nerve_wSe[0]],
            'nerve_class_sp': nerve_wSp[1] if len(nerve_wSp) > 1 else [nerve_wSp[0]],
            'cell_class_acc': cell_wAcc[1] if len(cell_wAcc) > 1 else [cell_wAcc[0]],
            'cell_class_se': cell_wSe[1] if len(cell_wSe) > 1 else [cell_wSe[0]],
            'cell_class_sp': cell_wSp[1] if len(cell_wSp) > 1 else [cell_wSp[0]]
        }
        
        return avg_loss, mean_acc, nerve_wAcc[0], cell_wAcc[0], metrics
    
    def evaluate(self, dataloader: DataLoader, save_predictions: bool = True, 
                 predictions_file: str = None, detailed: bool = True):
        """Documentation for this retained training component."""
        print("\n=== 评估双任务模型 ===")
        print("任务1: 角膜神经弯曲度分级")
        print("任务2: 朗格汉斯细胞活化程度分级")
        
        self.model.eval()
        
        all_nerve_probs = []
        all_cell_probs = []
        all_nerve_logits = []
        all_cell_logits = []
        all_nerve_labels = []
        all_cell_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                nerve_labels = batch['nerve_label'].to(self.device)
                cell_labels = batch['cell_label'].to(self.device)
                image_names = batch['image_name']
                
                nerve_logits, cell_logits = self.model(input_ids, attention_mask)
                nerve_probs = torch.softmax(nerve_logits, dim=1)
                cell_probs = torch.softmax(cell_logits, dim=1)
                
                all_nerve_probs.append(nerve_probs.cpu().numpy())
                all_cell_probs.append(cell_probs.cpu().numpy())
                all_nerve_logits.append(nerve_logits.cpu().numpy())
                all_cell_logits.append(cell_logits.cpu().numpy())
                all_nerve_labels.append(nerve_labels.cpu().numpy())
                all_cell_labels.append(cell_labels.cpu().numpy())
                all_image_names.extend(image_names)
        
        all_nerve_probs = np.concatenate(all_nerve_probs, axis=0)
        all_cell_probs = np.concatenate(all_cell_probs, axis=0)
        all_nerve_logits = np.concatenate(all_nerve_logits, axis=0)
        all_cell_logits = np.concatenate(all_cell_logits, axis=0)
        all_nerve_labels = np.concatenate(all_nerve_labels, axis=0)
        all_cell_labels = np.concatenate(all_cell_labels, axis=0)
        
        # Collect or process predictions.
        nerve_preds = np.argmax(all_nerve_probs, axis=1)
        cell_preds = np.argmax(all_cell_probs, axis=1)
        
        # Compute the evaluation metrics.
        nerve_wAcc, nerve_wSe, nerve_wSp = get_metrix(nerve_preds, all_nerve_labels)
        cell_wAcc, cell_wSe, cell_wSp = get_metrix(cell_preds, all_cell_labels)
        
        # Compute the required value.
        mean_acc = (nerve_wAcc[0] + cell_wAcc[0]) / 2
        
        # Collect or process predictions.
        if save_predictions and predictions_file:
            predictions_data = []
            for i in range(len(all_image_names)):
                predictions_data.append({
                    'name': all_image_names[i],
                    'text_nerve_prediction_probs': all_nerve_probs[i].tolist(),
                    'text_cell_prediction_probs': all_cell_probs[i].tolist(),
                    'true_nerve_label': int(all_nerve_labels[i]),
                    'true_cell_label': int(all_cell_labels[i])
                })
            
            # Ensure the output directory exists.
            os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            print(f"✓ 双任务预测结果已保存到: {predictions_file}")
            print(f"  保存了 {len(predictions_data)} 条预测记录")
        
        if detailed:
            # Report the current status.
            print("\n" + "=" * 70)
            print("双任务评估结果详细报告")
            print("=" * 70)
            
            print(f"\n总体指标:")
            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            table.add_row(["样本总数", len(all_nerve_labels)])
            table.add_row(["平均准确率", f"{mean_acc:.4f}"])
            table.add_row(["任务1准确率（神经弯曲度）", f"{nerve_wAcc[0]:.4f}"])
            table.add_row(["任务2准确率（朗格汉斯细胞）", f"{cell_wAcc[0]:.4f}"])
            print(table)
            
            # Retain this implementation detail from the original training pipeline.
            print(f"\n任务1 - 神经弯曲度分级详细指标:")
            nerve_levels = ["0级（无弯曲）", "1级（轻度弯曲）", "2级（中度弯曲）", "3级（重度弯曲）"]
            
            if len(nerve_wAcc) > 1 and len(nerve_wSe) > 1 and len(nerve_wSp) > 1:
                nerve_table = PrettyTable()
                nerve_table.field_names = ["弯曲度等级", "准确率", "敏感度", "特异度"]
                for i in range(self.num_classes):
                    nerve_table.add_row([
                        nerve_levels[i],
                        f"{nerve_wAcc[1][i]:.4f}",
                        f"{nerve_wSe[1][i]:.4f}",
                        f"{nerve_wSp[1][i]:.4f}"
                    ])
                print(nerve_table)
            
            # Retain this implementation detail from the original training pipeline.
            print(f"\n任务2 - 朗格汉斯细胞活化程度详细指标:")
            cell_levels = ["0级（无活化）", "1级（轻度活化）", "2级（中度活化）", "3级（重度活化）"]
            
            if len(cell_wAcc) > 1 and len(cell_wSe) > 1 and len(cell_wSp) > 1:
                cell_table = PrettyTable()
                cell_table.field_names = ["活化程度", "准确率", "敏感度", "特异度"]
                for i in range(self.num_classes):
                    cell_table.add_row([
                        cell_levels[i],
                        f"{cell_wAcc[1][i]:.4f}",
                        f"{cell_wSe[1][i]:.4f}",
                        f"{cell_wSp[1][i]:.4f}"
                    ])
                print(cell_table)
        
        # Retain this implementation detail from the original training pipeline.
        results = {
            'mean_acc': mean_acc,
            'nerve_wAcc': nerve_wAcc,
            'nerve_wSe': nerve_wSe,
            'nerve_wSp': nerve_wSp,
            'cell_wAcc': cell_wAcc,
            'cell_wSe': cell_wSe,
            'cell_wSp': cell_wSp,
            'num_samples': len(all_nerve_labels)
        }
        
        return results, all_nerve_probs, all_cell_probs, all_nerve_labels, all_cell_labels, nerve_preds, cell_preds, all_image_names


# Run the evaluation step.
def main(args):
    """Documentation for this retained training component."""
    
    # Set the random seed for reproducibility.
    set_seed(args.seed)
    
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
    
    # Retain this implementation detail from the original training pipeline.
    use_pretrained_encoder = args.use_pretrained_encoder
    freeze_encoder = args.freeze_encoder
    
    try:
        print("\n" + "=" * 80)
        print("双任务四分类任务：角膜神经弯曲度分级 + 朗格汉斯细胞活化程度分级")
        print("任务1: 神经弯曲度 - 0级（无弯曲）, 1级（轻度弯曲）, 2级（中度弯曲）, 3级（重度弯曲）")
        print("任务2: 朗格汉斯细胞 - 0级（无活化）, 1级（轻度活化）, 2级（中度活化）, 3级（重度活化）")
        print("=" * 80)
        
        # Load or validate the configuration.
        print("\n参数配置:")
        print(f"  训练数据: {args.train_json}")
        print(f"  验证数据: {args.val_json}")
        print(f"  基础模型: {args.base_model}")
        print(f"  分词器路径: {args.tokenizer_path if args.tokenizer_path else '与模型路径相同'}")
        print(f"  使用慢速分词器: {args.use_slow_tokenizer}")
        print(f"  使用预训练编码器: {'是' if use_pretrained_encoder else '否'}")
        if use_pretrained_encoder:
            print(f"  预训练生成模型: {args.pretrained_generative_model}")
        print(f"  冻结编码器: {'是' if freeze_encoder else '否'}")
        print(f"  预测结果保存: {args.predictions_file}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  训练轮数: {args.num_epochs}")
        print(f"  批次大小: {args.batch_size}")
        print(f"  特征维度: {args.feature_dim}")
        print(f"  学习率: {args.learning_rate}")
        
        # Configure or use the model.
        print("\n步骤1: 初始化tokenizer")
        tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=False,
            use_fast=not args.use_slow_tokenizer
        )
        print(f"✓ Tokenizer加载完成，词汇表大小: {tokenizer.vocab_size}")
        print(f"  分词器路径: {tokenizer_path}")
        print(f"  使用快速分词器: {not args.use_slow_tokenizer}")
        
        # Run the training step.
        print("\n步骤2: 加载训练数据")
        train_dataset = DualTaskJsonDataset(args.train_json, tokenizer, max_len=args.max_len)
        
        # Run the training step.
        nerve_labels_array = np.array(train_dataset.nerve_labels)
        cell_labels_array = np.array(train_dataset.cell_labels)
        num_train_samples = len(train_dataset)
        
        print(f"\n训练集统计:")
        print(f"  总样本数: {num_train_samples}")
        
        nerve_class_counts = np.bincount(nerve_labels_array, minlength=4)
        cell_class_counts = np.bincount(cell_labels_array, minlength=4)
        
        print(f"\n任务1（神经弯曲度）分布:")
        for i in range(4):
            count = nerve_class_counts[i]
            ratio = count / num_train_samples
            print(f"  等级 {i}: {count}个样本 ({ratio:.2%})")
        
        print(f"\n任务2（朗格汉斯细胞）分布:")
        for i in range(4):
            count = cell_class_counts[i]
            ratio = count / num_train_samples
            print(f"  等级 {i}: {count}个样本 ({ratio:.2%})")
        
        # Run the training step.
        print("\n步骤3: 训练最终模型")
        
        # Prepare or inspect the dataset.
        print("\n加载验证数据集...")
        val_dataset = DualTaskJsonDataset(args.val_json, tokenizer, max_len=args.max_len)
        
        # Run the validation step.
        val_nerve_labels_array = np.array(val_dataset.nerve_labels)
        val_cell_labels_array = np.array(val_dataset.cell_labels)
        num_val_samples = len(val_dataset)
        
        print(f"\n验证集统计:")
        print(f"  总样本数: {num_val_samples}")
        
        val_nerve_class_counts = np.bincount(val_nerve_labels_array, minlength=4)
        val_cell_class_counts = np.bincount(val_cell_labels_array, minlength=4)
        
        print(f"\n任务1（神经弯曲度）分布:")
        for i in range(4):
            count = val_nerve_class_counts[i]
            ratio = count / num_val_samples
            print(f"  等级 {i}: {count}个样本 ({ratio:.2%})")
        
        print(f"\n任务2（朗格汉斯细胞）分布:")
        for i in range(4):
            count = val_cell_class_counts[i]
            ratio = count / num_val_samples
            print(f"  等级 {i}: {count}个样本 ({ratio:.2%})")
        
        # Create the required object.
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"最终训练集: {len(train_dataset)} 个样本")
        print(f"最终验证集: {len(val_dataset)} 个样本")
        
        # Create the required object.
        final_classifier = DualTaskClassifier(
            base_model_path=args.base_model,
            tokenizer_path=args.tokenizer_path,
            feature_dim=args.feature_dim,
            num_classes=4,
            device=device,
            use_slow_tokenizer=args.use_slow_tokenizer
        )
        
        # Run the training step.
        print("\n训练最终模型...")
        history = final_classifier.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            freeze_encoder=freeze_encoder,
            use_pretrained_encoder=use_pretrained_encoder,
            pretrained_generative_model_path=args.pretrained_generative_model,
            output_dir=args.output_dir
        )
        
        # Run the evaluation step.
        print("\n步骤4: 最终评估并保存预测结果")
        final_results, nerve_probs, cell_probs, nerve_labels, cell_labels, nerve_preds, cell_preds, image_names = final_classifier.evaluate(
            val_loader, 
            save_predictions=True,
            predictions_file=args.predictions_file,
            detailed=True
        )
        
        # Configure or use the model.
        print("\n步骤5: 保存模型和训练结果")
        
        # Ensure the output directory exists.
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure or use the model.
        model_weights_file = f"{args.output_dir}/model_weights_final.pth"
        torch.save(final_classifier.model.state_dict(), model_weights_file)
        # Configure or use the model.
        final_classifier._cleanup_old_models()
        
        # Configure or use the model.
        model_config = {
            'model_type': 'T5EncoderForDualTasks',
            'base_model_path': args.base_model,
            'tokenizer_path': args.tokenizer_path if args.tokenizer_path else args.base_model,
            'feature_dim': args.feature_dim,
            'num_classes': 4,
            'nerve_class_names': train_dataset.nerve_class_names,
            'cell_class_names': train_dataset.cell_class_names,
            'vocab_size': tokenizer.vocab_size,
            'max_length': args.max_len,
            'use_pretrained_encoder': use_pretrained_encoder,
            'pretrained_generative_model_path': args.pretrained_generative_model,
            'freeze_encoder': freeze_encoder,
            'use_slow_tokenizer': args.use_slow_tokenizer,
            'training_config': {
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'device': str(device),
                'seed': args.seed
            }
        }
        
        config_file = f"{args.output_dir}/config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, ensure_ascii=False, indent=2)
        
        # Run the training step.
        all_results = {
            'task_info': model_config,
            'dataset_info': {
                'train_samples': num_train_samples,
                'val_samples': num_val_samples,
                'train_nerve_label_distribution': nerve_class_counts.tolist(),
                'train_cell_label_distribution': cell_class_counts.tolist(),
                'val_nerve_label_distribution': val_nerve_class_counts.tolist(),
                'val_cell_label_distribution': val_cell_class_counts.tolist(),
                'train_json_path': args.train_json,
                'val_json_path': args.val_json
            },
            'final_results': final_results,
            'predictions_info': {
                'predictions_file': args.predictions_file,
                'num_predictions': len(image_names)
            }
        }
        
        results_file = f"{args.output_dir}/dual_task_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"\n✓ 所有结果已保存至目录: {args.output_dir}")
        print(f"  模型权重: {model_weights_file}")
        print(f"  模型配置: {config_file}")
        print(f"  训练结果: {results_file}")
        print(f"  预测结果: {args.predictions_file}")
        
        # Retain this implementation detail from the original training pipeline.
        print("\n" + "=" * 70)
        print("双任务分类完成总结")
        print("=" * 70)
        
        print(f"\n训练配置:")
        print(f"  使用预训练编码器: {'是' if use_pretrained_encoder else '否'}")
        if use_pretrained_encoder:
            print(f"  预训练模型路径: {args.pretrained_generative_model}")
        print(f"  冻结编码器: {'是' if freeze_encoder else '否'}")
        
        print(f"\n性能指标:")
        print(f"  平均准确率: {final_results['mean_acc']:.4f}")
        print(f"  任务1（神经弯曲度）准确率: {final_results['nerve_wAcc'][0]:.4f}")
        print(f"  任务2（朗格汉斯细胞）准确率: {final_results['cell_wAcc'][0]:.4f}")
        
        print(f"\n✓ 双任务分类流程已完成！")
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


# Retain this implementation detail from the original training pipeline.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="双任务四分类模型训练（神经弯曲度 + 朗格汉斯细胞活化程度）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Resolve the required path.
    parser.add_argument('--train_json', type=str,
                    default="/data/Desktop/BioMiner/Generative_model/datasets/CORN_LCs/Train_data/CCM_finetune_train.json",
                    help='训练数据JSON文件路径')
    
    parser.add_argument('--val_json', type=str,
                    default="/data/Desktop/BioMiner/Generative_model/datasets/CORN_LCs/Train_data/CCM_finetune_val.json",
                    help='验证数据JSON文件路径')
    
    # Configure or use the model.
    parser.add_argument('--base_model', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/models/t5-clinical-base",
                       help='基础T5模型路径')
    
    parser.add_argument('--tokenizer_path', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer_CNs_LCs/",
                       help='分词器路径（如果与模型路径不同，默认: 使用base_model路径）')
    
    parser.add_argument('--pretrained_generative_model', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/checkpoint/Generative_model/best_model",
                       help='预训练生成模型路径（当使用预训练编码器时需要）')
    
    # Resolve the required path.
    parser.add_argument('--predictions_file', type=str,
                       default="/data/Desktop/BioMiner/Generative_model/text_grading_predictions_CNs_LCs.json",
                       help='预测结果保存路径')
    
    parser.add_argument('--output_dir', type=str,
                       default="/data/Desktop/BioMiner/Generative_model/checkpoint/Finetune_model",
                       help='输出目录（训练结果和模型权重保存位置）')
    
    # Run the training step.
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数（默认: 50）')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小（默认: 16）')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率（默认: 1e-4）')
    
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='特征维度（默认: 256）')
    
    parser.add_argument('--max_len', type=int, default=512,
                       help='最大序列长度（默认: 512）')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    
    # Configure or use the model.
    parser.add_argument('--use_pretrained_encoder', action='store_true', default=True,
                       help='是否使用预训练的生成模型编码器权重（默认: True）')
    
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                       help='是否冻结编码器，只训练分类头（迁移学习模式）（默认: False）')
    
    parser.add_argument('--use_slow_tokenizer', action='store_true', default=False,
                       help='是否使用慢速分词器（默认: False）')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='运行设备（auto:自动选择GPU如果可用, cuda:强制GPU, cpu:强制CPU）（默认: auto）')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("双任务四分类模型训练（神经弯曲度 + 朗格汉斯细胞活化程度）")
    print("=" * 80)
    
    # Resolve the required path.
    required_paths = [
        ('训练数据', args.train_json),
        ('验证数据', args.val_json),
        ('基础模型', args.base_model)
    ]
    
    # Resolve the required path.
    missing_paths = []
    for name, path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        print("\n❌ 以下路径不存在，请检查:")
        for missing in missing_paths:
            print(f"  {missing}")
        print("\n请确保提供正确的路径参数")
        exit(1)
    
    # Run the training step.
    if args.use_pretrained_encoder and not args.pretrained_generative_model:
        print("\n❌ 错误: 使用 --use_pretrained_encoder 时必须提供 --pretrained_generative_model 路径")
        exit(1)
    
    if args.use_pretrained_encoder and not os.path.exists(args.pretrained_generative_model):
        print(f"\n❌ 错误: 预训练生成模型路径不存在: {args.pretrained_generative_model}")
        exit(1)
    
    # Configure or apply the tokenizer.
    if args.tokenizer_path and not os.path.exists(args.tokenizer_path):
        print(f"\n⚠ 警告: 指定的分词器路径不存在: {args.tokenizer_path}")
        print("  将使用基础模型路径作为分词器路径")
        args.tokenizer_path = ""
    
    # Retain this implementation detail from the original training pipeline.
    main(args)