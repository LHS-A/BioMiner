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
from prettytable import PrettyTable  # Added: For aesthetically formatted table printing
warnings.filterwarnings('ignore')

# Add project path to import evaluation metrics
sys.path.append(r"/data/Desktop/BioMiner")
from utils.evaluation_metrics import get_metrix


# --------------------- Set Random Seed ---------------------
def set_seed(seed: int = 42):
    """Set random seed to ensure reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# --------------------- JSON Dataset Class ---------------------
class LangerhansJsonDataset(Dataset):
    """Dataset class for Langerhans cell activation level four-class classification loaded from JSON files"""
    
    def __init__(self, json_path: str, tokenizer: T5Tokenizer, max_len: int = 512):
        """
        Initialize dataset
        
        Args:
            json_path: JSON file path
            tokenizer: T5 tokenizer
            max_len: Maximum sequence length
        """
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.labels = []  # Activation level labels: 0, 1, 2, 3
        self.texts = []
        self.image_names = []  # Image filenames (if available)
        
        print(f"Loading data from JSON file: {json_path}")
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON file should contain a list")
            
            for i, item in enumerate(tqdm(data, desc="Loading samples")):
                try:
                    # Check required fields
                    if not isinstance(item, dict):
                        print(f"Warning: Item {i} is not a dictionary format, skipping")
                        continue
                    
                    if 'label' not in item or 'input' not in item:
                        print(f"Warning: Item {i} missing label or input field, skipping")
                        continue
                    
                    # Get label
                    label = item['label']
                    if not isinstance(label, int):
                        try:
                            label = int(label)
                        except:
                            print(f"Warning: Label of item {i} is not an integer: {label}, skipping")
                            continue
                    
                    if label not in [0, 1, 2, 3]:
                        print(f"Warning: Label {label} of item {i} is not in valid range [0,1,2,3], skipping")
                        continue
                    
                    # Get text
                    text = item['input']
                    if not text or not isinstance(text, str):
                        print(f"Warning: Text of item {i} is empty or not a string, skipping")
                        continue
                    
                    # Get image filename (if exists) - Modified here: changed to get from 'name' field
                    image_name = item.get('name', f"sample_{i}")  # Modified: Get from 'name' field, not 'image_name'
                    
                    self.samples.append(item)
                    self.labels.append(label)
                    self.texts.append(text)
                    self.image_names.append(image_name)
                    
                except Exception as e:
                    print(f"Warning: Error processing item {i}: {e}, skipping")
                    continue
            
            if len(self.samples) == 0:
                raise ValueError(f"No valid samples found in {json_path}")
            
            print(f"✓ Loaded {len(self.samples)} samples")
            self._analyze_label_distribution()
            
        except Exception as e:
            raise ValueError(f"Failed to read JSON file: {e}")
    
    def _analyze_label_distribution(self):
        """Analyze label distribution"""
        labels_array = np.array(self.labels)
        num_samples = len(labels_array)
        
        print(f"\nDataset statistics:")
        print(f"Total samples: {num_samples}")
        
        # Label distribution
        activation_levels = ["Level 0 (no activation)", "Level 1 (mild activation)", "Level 2 (moderate activation)", "Level 3 (severe activation)"]
        print(f"\nActivation level distribution:")
        
        class_counts = np.bincount(labels_array, minlength=4)
        for level in range(4):
            count = class_counts[level]
            ratio = count / num_samples
            print(f"  {activation_levels[level]}: {count} samples ({ratio:.2%})")
        
        # Calculate class imbalance ratio
        max_count = np.max(class_counts)
        min_count = np.min(class_counts[class_counts > 0]) if np.any(class_counts > 0) else 0
        
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        
        # Save statistical information
        self.class_counts = class_counts
        self.class_names = activation_levels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
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
            'labels': label,
            'image_name': self.image_names[idx]
        }


# --------------------- T5 Encoder Model (Four-class Classification) Revised Version ---------------------
class T5EncoderForActivationLevel(nn.Module):
    def __init__(self, base_model_path: str, tokenizer_path: str = None, feature_dim: int = 256, 
                 num_classes: int = 4, use_pretrained_encoder: bool = True, 
                 pretrained_generative_model_path: str = None, use_slow_tokenizer: bool = True):
        """
        Initialize T5 encoder classification model
        
        Args:
            base_model_path: Base T5 model path
            tokenizer_path: Tokenizer path (if different from model path)
            feature_dim: Feature dimension
            num_classes: Number of classes
            use_pretrained_encoder: Whether to use pre-trained encoder weights
            pretrained_generative_model_path: Path to pre-trained generative model
            use_slow_tokenizer: Whether to use slow tokenizer
        """
        super().__init__()
        
        # Load base T5 model using the same method as the generative model
        print(f"Loading base T5 model: {base_model_path}")
        self.t5 = T5Model.from_pretrained(
            base_model_path,
            trust_remote_code=False
        )
        encoder_dim = self.t5.config.d_model
        
        # Feature extraction layers
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
        
        # Four-class classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Freeze decoder (only use encoder)
        for param in self.t5.decoder.parameters():
            param.requires_grad = False
        
        print(f"✓ Initialized T5 encoder + {num_classes}-class head (Langerhans cell activation level)")
        print(f"  Encoder dimension: {encoder_dim}")
        print(f"  Feature dimension: {feature_dim}")
        
        # Load pre-trained generative model weights if needed
        if use_pretrained_encoder and pretrained_generative_model_path:
            self._load_generative_model_weights(pretrained_generative_model_path)
    
    def _load_generative_model_weights(self, generative_model_path: str):
        """
        Load encoder weights from pre-trained generative model
        
        Args:
            generative_model_path: Saved path of generative model
        """
        print(f"\nLoading encoder weights from generative model: {generative_model_path}")
        
        if not os.path.exists(generative_model_path):
            print(f"Warning: Generative model path does not exist: {generative_model_path}")
            return
        
        try:
            # Load complete generative model
            generative_model = T5ForConditionalGeneration.from_pretrained(
                generative_model_path,
                trust_remote_code=False
            )
            
            # Get encoder state dictionary from generative model
            generative_encoder_state_dict = generative_model.encoder.state_dict()
            
            # Get current model's encoder state dictionary
            current_encoder_state_dict = self.t5.encoder.state_dict()
            
            # Calculate number of matched layers
            matched_layers = 0
            total_layers = len(current_encoder_state_dict)
            
            # Load weights layer by layer
            for key in current_encoder_state_dict:
                if key in generative_encoder_state_dict:
                    # Check if shapes match
                    if current_encoder_state_dict[key].shape == generative_encoder_state_dict[key].shape:
                        current_encoder_state_dict[key] = generative_encoder_state_dict[key].clone()
                        matched_layers += 1
                    else:
                        print(f"Warning: Layer {key} shape mismatch, skipping")
                        print(f"  Current shape: {current_encoder_state_dict[key].shape}")
                        print(f"  Generative model shape: {generative_encoder_state_dict[key].shape}")
            
            # Update encoder weights
            self.t5.encoder.load_state_dict(current_encoder_state_dict)
            
            print(f"✓ Successfully loaded encoder weights: {matched_layers}/{total_layers} layers matched")
            
        except Exception as e:
            print(f"❌ Failed to load generative model weights: {e}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get encoder output
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Average pooling (considering attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask
        
        # Feature projection and classification
        features = self.projection(pooled)
        logits = self.classifier(features)
        
        return logits


# --------------------- Classifier (Langerhans Cell Activation Level Four-class Classification) Revised Version ---------------------
class LangerhansActivationClassifier:
    def __init__(self, base_model_path: str, tokenizer_path: str = None, feature_dim: int = 256, 
                 num_classes: int = 4, device: str = 'cuda', use_pretrained_encoder: bool = False,
                 pretrained_generative_model_path: str = None, use_slow_tokenizer: bool = True):
        """
        Initialize classifier
        
        Args:
            base_model_path: Base model path
            tokenizer_path: Tokenizer path (if different from model path)
            feature_dim: Feature dimension
            num_classes: Number of classes
            device: Device
            use_pretrained_encoder: Whether to use pre-trained encoder
            pretrained_generative_model_path: Path to pre-trained generative model
            use_slow_tokenizer: Whether to use slow tokenizer
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.base_model_path = base_model_path
        self.tokenizer_path = tokenizer_path
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_slow_tokenizer = use_slow_tokenizer
        
        print(f"✓ Langerhans cell activation level classifier initialized")
        print(f"  Device: {self.device}")
        print(f"  Base model path: {base_model_path}")
        print(f"  Tokenizer path: {tokenizer_path if tokenizer_path else 'Same as model path'}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Use slow tokenizer: {use_slow_tokenizer}")
        
        # Model will be initialized later in fit method, as tokenizer needs to be initialized first
        
    def _init_model(self, use_pretrained_encoder: bool, pretrained_generative_model_path: str):
        """Initialize model (called after tokenizer initialization)"""
        self.model = T5EncoderForActivationLevel(
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
            print(f"  Loaded generative model encoder weights: {pretrained_generative_model_path}")
    
    def _save_predictions(self, dataloader: DataLoader, epoch: int, wacc: float):
        """Save prediction results to specified path (new method)"""
        predictions_file = "/data/Desktop/BioMiner/Generative_model/text_grading_predictions.json"
        print(f"\nSaving best model's prediction results to: {predictions_file}")
        
        self.model.eval()
        all_probs = []
        all_logits = []
        all_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating prediction results"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                image_names = batch['image_name']
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_image_names.extend(image_names)
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Get predicted classes
        preds = np.argmax(all_probs, axis=1)
        
        # Save prediction results - Modified: directly use image_names obtained from data as name field
        predictions_data = []
        for i in range(len(all_image_names)):
            predictions_data.append({
                'name': all_image_names[i],  # Directly use name obtained from data
                'text_prediction_probs': all_probs[i].tolist(),
                # 'predicted_label': int(preds[i]),
                'true_label': int(all_labels[i]),
                # 'epoch': epoch,
                # 'wacc': wacc
            })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Prediction results saved to: {predictions_file}")
        print(f"  Saved {len(predictions_data)} prediction records (from epoch {epoch}, wAcc: {wacc:.4f})")

    def _cleanup_old_models(self, keep_num: int = 4):
        """
        Clean up old model files, keep only the latest few
        
        Args:
            keep_num: Number of latest files to keep
        """
        if not hasattr(self, 'output_dir') or not self.output_dir:
            return
        
        try:
            # Get all .pth files in directory
            model_files = []
            for file in os.listdir(self.output_dir):
                if file.endswith('.pth'):
                    file_path = os.path.join(self.output_dir, file)
                    if os.path.isfile(file_path):
                        # Get file modification time
                        mtime = os.path.getmtime(file_path)
                        model_files.append((file_path, mtime))
            
            # If number of files exceeds keep_num
            if len(model_files) > keep_num:
                # Sort by modification time (latest first)
                model_files.sort(key=lambda x: x[1], reverse=True)
                
                # Keep latest keep_num files
                files_to_keep = [f[0] for f in model_files[:keep_num]]
                
                # Delete old files
                for file_path, _ in model_files[keep_num:]:
                    try:
                        os.remove(file_path)
                        print(f"✓ Deleted old model file: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"⚠ Failed to delete file {file_path}: {e}")
                
                print(f"✓ Cleaned up old model files, kept latest {keep_num} files")
        
        except Exception as e:
            print(f"⚠ Error cleaning up model files: {e}")

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None,
            num_epochs: int = 30, lr: float = 1e-4, weight_method='inverse',
            freeze_encoder: bool = False, use_pretrained_encoder: bool = False,
            pretrained_generative_model_path: str = None, output_dir: str = None):
        """Train model"""
        # Initialize model
        self._init_model(use_pretrained_encoder, pretrained_generative_model_path)
        self.model.to(self.device)
        
        # Save output directory
        self.output_dir = output_dir
        
        # Freeze encoder if needed
        if freeze_encoder:
            for param in self.model.t5.encoder.parameters():
                param.requires_grad = False
            print("✓ Freeze encoder parameters, only train classification head")
        
        # Calculate class weights
        if hasattr(train_dataloader.dataset, 'get_class_weights'):
            class_weights = train_dataloader.dataset.get_class_weights(method=weight_method)
            class_weights = class_weights.to(self.device)
            print(f"Using class weight calculation method: {weight_method}")
            print(f"Class weights: {class_weights.cpu().numpy()}")
        else:
            # Calculate class distribution from training set
            train_labels = []
            for batch in train_dataloader:
                train_labels.append(batch['labels'].numpy())
            
            if train_labels:
                train_labels_array = np.concatenate(train_labels, axis=0)
                class_counts = np.bincount(train_labels_array, minlength=self.num_classes)
                total_samples = len(train_labels_array)
                
                # Calculate inverse frequency weights
                class_weights = total_samples / (self.num_classes * class_counts + 1e-9)
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                
                print(f"Training set class distribution: {class_counts.tolist()}")
                print(f"Class weights: {class_weights.cpu().numpy()}")
            else:
                class_weights = None
        
        # Get parameters that need training
        if freeze_encoder:
            # Only train classification head and projection layer
            trainable_params = list(self.model.projection.parameters()) + list(self.model.classifier.parameters())
            print(f"Training parameters: projection layer + classification head")
        else:
            # Train all parameters
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            print(f"Training parameters: encoder + projection layer + classification head")
        
        # Optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr, weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # Training history records
        history = {
            'train_loss': [], 'val_loss': [],
            'val_wAcc': [], 'val_wSe': [], 'val_wSp': []
        }
        
        best_val_wAcc = -1.0  # Use weighted accuracy as early stopping criterion
        best_epoch = 0
        patience_counter = 0
        patience = 10
        
        print(f"\n=== Start Training ({num_epochs} epochs) ===")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0.0
            train_steps = 0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                
                # Use weighted cross-entropy loss
                if class_weights is not None:
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                
                loss = loss_fn(logits, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = total_loss / train_steps
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_dataloader:
                val_loss, val_wAcc, val_wSe, val_wSp, val_metrics = self._evaluate_epoch(val_dataloader, class_weights)
                
                history['val_loss'].append(val_loss)
                history['val_wAcc'].append(val_wAcc)
                history['val_wSe'].append(val_wSe)
                history['val_wSp'].append(val_wSp)
                
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                
                # Print detailed validation metrics table (following format similar to reference code A)
                table = PrettyTable()
                table.field_names = ["wAcc", "wSe", "wSp"]
                table.add_row([f"{val_wAcc:.4f}", f"{val_wSe:.4f}", f"{val_wSp:.4f}"])
                print(table)
                
                # Early stopping decision (based on weighted accuracy)
                if val_wAcc > best_val_wAcc:
                    best_val_wAcc = val_wAcc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    
                    # Save best model checkpoint
                    checkpoint_path = f"{self.output_dir}/best_model_epoch_{best_epoch}_wacc_{best_val_wAcc:.4f}.pth"
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"★ Save best model: {checkpoint_path}")
                    # Clean up old model files, keep only latest 4
                    self._cleanup_old_models()

                    # Save prediction results (overwriting existing file)
                    self._save_predictions(val_dataloader, best_epoch, best_val_wAcc)
                    
                    # Print best model's detailed class metrics (following format similar to reference code A)
                    print("Best model class metrics:")
                    best_table = PrettyTable()
                    best_table.field_names = ["Class", "Accuracy", "Sensitivity", "Specificity"]
                    for i in range(len(val_metrics['class_acc'])):
                        best_table.add_row([
                            f"Class {i+1}", 
                            f"{val_metrics['class_acc'][i]:.4f}", 
                            f"{val_metrics['class_se'][i]:.4f}", 
                            f"{val_metrics['class_sp'][i]:.4f}"
                        ])
                    print(best_table)
                    print(f"Mean - Accuracy: {val_metrics['mean_acc']:.4f}, Sensitivity: {val_metrics['mean_se']:.4f}, Specificity: {val_metrics['mean_sp']:.4f}")
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch + 1} - Current best mean accuracy: {best_val_wAcc:.4f} (epoch {best_epoch})")
                    if patience_counter >= patience:
                        print(f"Early stopping triggered, best weighted accuracy: {best_val_wAcc:.4f}")
                        break
            else:
                print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}")
            
            scheduler.step()
        
        # Load best model
        if os.path.exists(f"{self.output_dir}/best_model_epoch_{best_epoch}_wacc_{best_val_wAcc:.4f}.pth"):
            self.model.load_state_dict(torch.load(
                f"{self.output_dir}/best_model_epoch_{best_epoch}_wacc_{best_val_wAcc:.4f}.pth",
                map_location=self.device
            ))
            print(f"★ Load best model (epoch {best_epoch}, wAcc: {best_val_wAcc:.4f})")
        
        return history
    
    def _evaluate_epoch(self, dataloader: DataLoader, class_weights=None):
        """Validation for single epoch (modified to return detailed metrics)"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                if class_weights is not None:
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                
                # Get prediction results
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Use imported get_metrix function to calculate weighted metrics
        wAcc_result, wSe_result, wSp_result = get_metrix(all_preds, all_labels)
        
        avg_loss = total_loss / len(dataloader)
        
        # Return detailed metrics (following format similar to reference code A)
        metrics = {
            'mean_acc': wAcc_result[0],
            'mean_se': wSe_result[0], 
            'mean_sp': wSp_result[0],
            'class_acc': wAcc_result[1] if len(wAcc_result) > 1 else [wAcc_result[0]],
            'class_se': wSe_result[1] if len(wSe_result) > 1 else [wSe_result[0]],
            'class_sp': wSp_result[1] if len(wSp_result) > 1 else [wSp_result[0]]
        }
        
        return avg_loss, wAcc_result[0], wSe_result[0], wSp_result[0], metrics
    
    def evaluate(self, dataloader: DataLoader, save_predictions: bool = True, 
                 predictions_file: str = None, detailed: bool = True):
        """Evaluate model and save prediction results"""
        print("\n=== Evaluate Model (Langerhans Cell Activation Level Classification) ===")
        self.model.eval()
        
        all_probs = []
        all_logits = []
        all_labels = []
        all_image_names = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                image_names = batch['image_name']
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_image_names.extend(image_names)
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Get predicted classes
        preds = np.argmax(all_probs, axis=1)
        
        # Use imported get_metrix function to calculate evaluation metrics
        wAcc_result, wSe_result, wSp_result = get_metrix(preds, all_labels)
        
        # Save prediction results - Modified: directly use image_names obtained from data as name field
        if save_predictions and predictions_file:
            predictions_data = []
            for i in range(len(all_image_names)):
                predictions_data.append({
                    'name': all_image_names[i],  # Directly use name obtained from data
                    'text_prediction_probs': all_probs[i].tolist(),
                    'true_label': int(all_labels[i])
                })
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            print(f"✓ Prediction results saved to: {predictions_file}")
            print(f"  Saved {len(predictions_data)} prediction records")
        
        if detailed:
            # Print detailed results (using PrettyTable format)
            print("\n" + "=" * 70)
            print("Detailed Evaluation Results Report")
            print("=" * 70)
            
            print(f"\nOverall metrics:")
            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            table.add_row(["Total samples", len(all_labels)])
            table.add_row(["Weighted Accuracy (wAcc)", f"{wAcc_result[0]:.4f}"])
            table.add_row(["Weighted Sensitivity (wSe)", f"{wSe_result[0]:.4f}"])
            table.add_row(["Weighted Specificity (wSp)", f"{wSp_result[0]:.4f}"])
            print(table)
            
            print(f"\nDetailed metrics by class:")
            activation_levels = ["Level 0 (no activation)", "Level 1 (mild activation)", "Level 2 (moderate activation)", "Level 3 (severe activation)"]
            
            if len(wAcc_result) > 1 and len(wSe_result) > 1 and len(wSp_result) > 1:
                class_table = PrettyTable()
                class_table.field_names = ["Activation Level", "Accuracy", "Sensitivity", "Specificity"]
                for i in range(self.num_classes):
                    class_table.add_row([
                        activation_levels[i],
                        f"{wAcc_result[1][i]:.4f}",
                        f"{wSe_result[1][i]:.4f}",
                        f"{wSp_result[1][i]:.4f}"
                    ])
                print(class_table)
        
        # Return results
        results = {
            'wAcc': wAcc_result,
            'wSe': wSe_result,
            'wSp': wSp_result,
            'num_samples': len(all_labels)
        }
        
        return results, all_probs, all_labels, preds, all_image_names


# --------------------- 5-Fold Cross Validation ---------------------
def kfold_cross_validation(dataset: LangerhansJsonDataset, model_path: str, tokenizer_path: str = None,
                           feature_dim: int = 256, n_splits: int = 5, num_epochs: int = 30, 
                           batch_size: int = 8, use_pretrained_encoder: bool = False,
                           pretrained_generative_model_path: str = None, freeze_encoder: bool = False,
                           use_slow_tokenizer: bool = True, output_dir: str = None):
    """5-Fold Cross Validation: Langerhans Cell Activation Level Classification"""
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_results = []
    
    print(f"\n=== Start {n_splits}-Fold Cross Validation ===")
    print(f"Feature dimension: {feature_dim}, Batch size: {batch_size}, Training epochs: {num_epochs}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), dataset.labels), 1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}/{n_splits}")
        print(f"{'=' * 60}")
        
        # Create training and validation sets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        # Check validation set label distribution
        val_labels = [dataset.labels[i] for i in val_idx]
        val_label_counts = np.bincount(val_labels, minlength=4)
        print(f"Validation set label distribution: {val_label_counts.tolist()}")
        
        # Train model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = LangerhansActivationClassifier(
            base_model_path=model_path,
            tokenizer_path=tokenizer_path,
            feature_dim=feature_dim,
            num_classes=4,
            device=device,
            use_slow_tokenizer=use_slow_tokenizer
        )
        
        history = classifier.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=num_epochs,
            lr=1e-4,
            freeze_encoder=freeze_encoder,
            use_pretrained_encoder=use_pretrained_encoder,
            pretrained_generative_model_path=pretrained_generative_model_path,
            output_dir=f"{output_dir}/fold_{fold}" if output_dir else None
        )
        
        # Evaluate (do not save prediction results)
        results, probs, labels, preds, image_names = classifier.evaluate(
            val_loader, 
            save_predictions=False,
            detailed=False
        )
        
        # Record results
        fold_result = {
            'fold': fold,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'val_label_distribution': val_label_counts.tolist(),
            'results': results,
            'history': {
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'final_val_wAcc': history['val_wAcc'][-1] if history['val_wAcc'] else None
            }
        }
        all_results.append(fold_result)
        
        print(f"\nFold {fold} results:")
        print(f"  Weighted Accuracy (wAcc): {results['wAcc'][0]:.4f}")
        print(f"  Weighted Sensitivity (wSe): {results['wSe'][0]:.4f}")
        print(f"  Weighted Specificity (wSp): {results['wSp'][0]:.4f}")
    
    # Aggregate results
    print(f"\n{'=' * 70}")
    print(f"{n_splits}-Fold Cross Validation Summary")
    print(f"{'=' * 70}")
    
    # Metrics to aggregate
    metrics_to_avg = ['wAcc', 'wSe', 'wSp']
    summary = {}
    
    for metric in metrics_to_avg:
        values = []
        for result in all_results:
            if metric in result['results']:
                values.append(result['results'][metric][0])
        
        if values:
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
    
    print("\nAverage metrics (± standard deviation) [minimum, maximum]:")
    for metric in metrics_to_avg:
        mean_key = f'{metric}_mean'
        if mean_key in summary:
            mean_val = summary[mean_key]
            std_val = summary[f'{metric}_std']
            min_val = summary[f'{metric}_min']
            max_val = summary[f'{metric}_max']
            
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]")
    
    return all_results, summary


# --------------------- Main Training and Evaluation Function ---------------------
def main(args):
    """Main function: Langerhans cell activation level classification task"""
    
    # Set random seed
    set_seed(args.seed)
    
    # Device selection
    if args.device == 'cuda':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print("Warning: Requested CUDA but CUDA not available, will use CPU")
    elif args.device == 'cpu':
        device = "cpu"
    else:  # 'auto'
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Parse boolean parameters
    use_pretrained_encoder = args.use_pretrained_encoder
    freeze_encoder = args.freeze_encoder
    
    try:
        print("\n" + "=" * 80)
        print("Corneal Langerhans Cell Activation Level Classification Task")
        print("Activation levels: Level 0 (no activation), Level 1 (mild activation), Level 2 (moderate activation), Level 3 (severe activation)")
        print("=" * 80)
        
        # Print parameter configuration
        print("\nParameter configuration:")
        print(f"  Training data: {args.train_json}")
        print(f"  Validation data: {args.val_json}")
        print(f"  Base model: {args.base_model}")
        print(f"  Tokenizer path: {args.tokenizer_path if args.tokenizer_path else 'Same as model path'}")
        print(f"  Use slow tokenizer: {args.use_slow_tokenizer}")
        print(f"  Use pre-trained encoder: {'Yes' if use_pretrained_encoder else 'No'}")
        if use_pretrained_encoder:
            print(f"  Pre-trained generative model: {args.pretrained_generative_model}")
        print(f"  Freeze encoder: {'Yes' if freeze_encoder else 'No'}")
        print(f"  Save prediction results: {args.predictions_file}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Training epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Feature dimension: {args.feature_dim}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Perform cross-validation: {'Yes' if args.do_cross_validation else 'No'}")
        
        # 1. Initialize tokenizer (using same method as generative model)
        print("\nStep 1: Initialize tokenizer")
        tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=False,
            use_fast=not args.use_slow_tokenizer
        )
        print(f"✓ Tokenizer loaded successfully, vocabulary size: {tokenizer.vocab_size}")
        print(f"  Tokenizer path: {tokenizer_path}")
        print(f"  Use fast tokenizer: {not args.use_slow_tokenizer}")
        
        # 2. Load training data
        print("\nStep 2: Load training data")
        train_dataset = LangerhansJsonDataset(args.train_json, tokenizer, max_len=args.max_len)
        
        # Analyze training data
        labels_array = np.array(train_dataset.labels)
        num_train_samples = len(train_dataset)
        
        print(f"\nTraining set statistics:")
        print(f"  Total samples: {num_train_samples}")
        
        class_counts = np.bincount(labels_array, minlength=4)
        activation_levels = train_dataset.class_names
        
        for i in range(4):
            count = class_counts[i]
            ratio = count / num_train_samples
            print(f"  {activation_levels[i]}: {count} samples ({ratio:.2%})")
        
        # 3. Perform cross-validation (optional)
        if args.do_cross_validation:
            print("\nStep 3: Perform cross-validation")
            kfold_results, kfold_summary = kfold_cross_validation(
                dataset=train_dataset,
                model_path=args.base_model,
                tokenizer_path=args.tokenizer_path,
                feature_dim=args.feature_dim,
                n_splits=args.n_splits,
                num_epochs=args.cv_epochs,
                batch_size=args.batch_size,
                use_pretrained_encoder=use_pretrained_encoder,
                pretrained_generative_model_path=args.pretrained_generative_model,
                freeze_encoder=freeze_encoder,
                use_slow_tokenizer=args.use_slow_tokenizer,
                output_dir=args.output_dir
            )
        else:
            kfold_results = None
            kfold_summary = None
        
        # 4. Train final model
        print("\nStep 4: Train final model")
        
        # Load validation dataset
        print("\nLoading validation dataset...")
        val_dataset = LangerhansJsonDataset(args.val_json, tokenizer, max_len=args.max_len)
        
        # Analyze validation data
        val_labels_array = np.array(val_dataset.labels)
        num_val_samples = len(val_dataset)
        
        print(f"\nValidation set statistics:")
        print(f"  Total samples: {num_val_samples}")
        
        val_class_counts = np.bincount(val_labels_array, minlength=4)
        for i in range(4):
            count = val_class_counts[i]
            ratio = count / num_val_samples
            print(f"  {activation_levels[i]}: {count} samples ({ratio:.2%})")
        
        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Final training set: {len(train_dataset)} samples")
        print(f"Final validation set: {len(val_dataset)} samples")
        
        # Create classifier
        final_classifier = LangerhansActivationClassifier(
            base_model_path=args.base_model,
            tokenizer_path=args.tokenizer_path,
            feature_dim=args.feature_dim,
            num_classes=4,
            device=device,
            use_slow_tokenizer=args.use_slow_tokenizer
        )
        
        # Train model
        print("\nTraining final model...")
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
        
        # 5. Final evaluation and save prediction results (this is automatically done in fit method)
        print("\nStep 5: Final evaluation and save prediction results")
        final_results, probs, labels, preds, image_names = final_classifier.evaluate(
            val_loader, 
            save_predictions=True,
            predictions_file=args.predictions_file,
            detailed=True
        )
        
        # 6. Save model and results
        print("\nStep 6: Save model and training results")
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model weights (best model already saved in fit method)
        model_weights_file = f"{args.output_dir}/model_weights_final.pth"
        torch.save(final_classifier.model.state_dict(), model_weights_file)
        # Clean up output directory, keep only latest 4 model files
        final_classifier._cleanup_old_models()
        
        # Save model configuration
        model_config = {
            'model_type': 'T5EncoderForActivationLevel',
            'base_model_path': args.base_model,
            'tokenizer_path': args.tokenizer_path if args.tokenizer_path else args.base_model,
            'feature_dim': args.feature_dim,
            'num_classes': 4,
            'class_names': activation_levels,
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
        
        # Save training results
        all_results = {
            'task_info': model_config,
            'dataset_info': {
                'train_samples': num_train_samples,
                'val_samples': num_val_samples,
                'train_label_distribution': class_counts.tolist(),
                'val_label_distribution': val_class_counts.tolist(),
                'train_json_path': args.train_json,
                'val_json_path': args.val_json
            },
            'final_results': final_results,
            'kfold_summary': kfold_summary,
            'predictions_info': {
                'predictions_file': args.predictions_file,
                'num_predictions': len(image_names)
            }
        }
        
        results_file = f"{args.output_dir}/classification_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"\n✓ All results saved to directory: {args.output_dir}")
        print(f"  Model weights: {model_weights_file}")
        print(f"  Model configuration: {config_file}")
        print(f"  Training results: {results_file}")
        print(f"  Prediction results: {args.predictions_file}")
        
        # 7. Final summary
        print("\n" + "=" * 70)
        print("Task Completion Summary")
        print("=" * 70)
        
        print(f"\nTraining configuration:")
        print(f"  Use pre-trained encoder: {'Yes' if use_pretrained_encoder else 'No'}")
        if use_pretrained_encoder:
            print(f"  Pre-trained model path: {args.pretrained_generative_model}")
        print(f"  Freeze encoder: {'Yes' if freeze_encoder else 'No'}")
        
        print(f"\nPerformance metrics:")
        print(f"  Weighted Accuracy (wAcc): {final_results['wAcc'][0]:.4f}")
        print(f"  Weighted Sensitivity (wSe): {final_results['wSe'][0]:.4f}")
        print(f"  Weighted Specificity (wSp): {final_results['wSp'][0]:.4f}")
        
        if args.do_cross_validation and kfold_summary:
            print(f"\nCross-validation stability:")
            print(f"  wAcc range: {kfold_summary.get('wAcc_min', 0):.4f} - {kfold_summary.get('wAcc_max', 0):.4f}")
            print(f"  wAcc mean ± standard deviation: {kfold_summary.get('wAcc_mean', 0):.4f} ± {kfold_summary.get('wAcc_std', 0):.4f}")
        
        print(f"\n✓ All processes completed!")
        
    except Exception as e:
        print(f"\n❌ Program execution error: {e}")
        import traceback
        traceback.print_exc()


# --------------------- Command Line Argument Parser (Improved Version) ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Langerhans Cell Activation Level Classification Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # ================ Data Path Parameters ================
    parser.add_argument('--train_json', type=str,
                    default="/data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data/LCs_finetune_train.json",
                    help='Training data JSON file path (default: /data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data/LCs_finetune_train.json)')

    parser.add_argument('--val_json', type=str,
                    default="/data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data/LCs_finetune_test.json",
                    help='Validation data JSON file path (default: /data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data/LCs_finetune_test.json)')
        
    # ================ Model Path Parameters ================
    parser.add_argument('--base_model', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/models/t5-clinical-base",
                       help='Base T5 model path (default: /data/Desktop/BioMiner/Generative_model/models/t5-clinical-base)')
    
    parser.add_argument('--tokenizer_path', type=str, default="/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer/",
                       help='Tokenizer path (if different from model path, default: use base_model path)')
    
    parser.add_argument('--pretrained_generative_model', type=str, default="/data/Desktop/BioMiner/Generative_model/checkpoint/Generative_model/best_model",
                       help='Pre-trained generative model path (required when using pre-trained encoder)')
    
    # ================ Output Path Parameters ================
    parser.add_argument('--predictions_file', type=str,
                       default="/data/Desktop/BioMiner/Generative_model/text_grading_predictions.json",
                       help='Prediction results save path (default: /data/Desktop/BioMiner/Generative_model/text_grading_predictions.json)')
    
    parser.add_argument('--output_dir', type=str,
                       default="/data/Desktop/BioMiner/Generative_model/checkpoint/Finetune_model",
                       help='Output directory (training results and model weights save location) (default: /data/Desktop/BioMiner/Generative_model/checkpoint/Finetune_model)')
    
    # ================ Training Parameters ================
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension (default: 256)')
    
    parser.add_argument('--max_len', type=int, default=512,
                       help='Maximum sequence length (default: 512)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # ================ Model Configuration Parameters (Improved Version, with defaults) ================
    parser.add_argument('--use_pretrained_encoder', action='store_true', default=True,
                       help='Whether to use pre-trained generative model encoder weights (default: True')
    
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                       help='Whether to freeze encoder, only train classification head (transfer learning mode) (default: False)')
    
    parser.add_argument('--use_slow_tokenizer', action='store_true', default=False,
                       help='Whether to use slow tokenizer (default: False')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Running device (auto: automatically select GPU if available, cuda: force GPU, cpu: force CPU) (default: auto)')
    
    # ================ Cross-Validation Parameters ================
    parser.add_argument('--do_cross_validation', action='store_true', default=False,
                       help='Whether to perform cross-validation (default: False)')
    
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    
    parser.add_argument('--cv_epochs', type=int, default=20,
                       help='Training epochs for cross-validation (default: 20)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("Langerhans Cell Activation Level Classification Model Training")
    print("=" * 80)
    
    # Check if necessary paths exist
    required_paths = [
        ('Training data', args.train_json),
        ('Validation data', args.val_json),
        ('Base model', args.base_model)
    ]
    
    # Check paths
    missing_paths = []
    for name, path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        print("\n❌ The following paths do not exist, please check:")
        for missing in missing_paths:
            print(f"  {missing}")
        print("\nPlease ensure correct path parameters are provided")
        exit(1)
    
    # Check pre-trained model path (if explicitly needed)
    if args.use_pretrained_encoder and not args.pretrained_generative_model:
        print("\n❌ Error: Must provide --pretrained_generative_model path when using --use_pretrained_encoder")
        exit(1)
    
    if args.use_pretrained_encoder and not os.path.exists(args.pretrained_generative_model):
        print(f"\n❌ Error: Pre-trained generative model path does not exist: {args.pretrained_generative_model}")
        exit(1)
    
    # Check tokenizer path
    if args.tokenizer_path and not os.path.exists(args.tokenizer_path):
        print(f"\n⚠ Warning: Specified tokenizer path does not exist: {args.tokenizer_path}")
        print("  Will use base model path as tokenizer path")
        args.tokenizer_path = ""
    
    # Run main function
    main(args)
