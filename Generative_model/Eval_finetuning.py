# This code is used to evaluate the model trained in Stage2_finetune.
# Simply load the weights from /data/Desktop/BioMiner/Generative_model/checkpoint/Finetune_model,
# specify the dataset (by modifying the two required parameters at the bottom), and run the script.

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5Model, T5ForConditionalGeneration
from tqdm import tqdm
import warnings
import sys
import argparse
from prettytable import PrettyTable
warnings.filterwarnings('ignore')

# Add project path to import evaluation metrics
sys.path.append(r"/data/Desktop/BioMiner")
from utils.evaluation_metrics import get_metrix

# --------------------- Set Random Seed ---------------------
def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --------------------- T5 Encoder Model for Activation Level (Four-class) - Fixed Version ---------------------
class T5EncoderForActivationLevel(nn.Module):
    def __init__(self, base_model_path: str, feature_dim: int = 256, num_classes: int = 4,
                 use_pretrained_encoder: bool = False, pretrained_generative_model_path: str = None):
        """
        Initialize T5 encoder classification model (consistent with Code A)
        
        Args:
            base_model_path: Base T5 model path
            feature_dim: Feature dimension
            num_classes: Number of classes
            use_pretrained_encoder: Whether to use pretrained encoder weights (retained for consistency, though not needed for evaluation)
            pretrained_generative_model_path: Path to the pretrained generative model
        """
        super().__init__()
        
        print(f"Loading base T5 model: {base_model_path}")
        # Load T5 model using the same method as Code A
        self.t5 = T5Model.from_pretrained(
            base_model_path,
            trust_remote_code=False
        )
        encoder_dim = self.t5.config.d_model
        
        # Feature extraction layers identical to Code A
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
        
        # Freeze decoder (use encoder only) - consistent with Code A
        for param in self.t5.decoder.parameters():
            param.requires_grad = False
        
        print(f"✓ Initialized T5 encoder + {num_classes}-class head (Langerhans cell activation level)")
        print(f"  Encoder dimension: {encoder_dim}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Decoder frozen: Yes")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get encoder outputs
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Mean pooling (considering attention mask) - identical to Code A
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask
        
        # Feature projection and classification
        features = self.projection(pooled)
        logits = self.classifier(features)
        
        return logits

# --------------------- JSON Dataset Class (Identical to Code A) ---------------------
class LangerhansJsonDataset(Dataset):
    """Load Langerhans cell activation level four-class dataset from JSON file"""
    
    def __init__(self, json_path: str, tokenizer, max_len: int = 512):
        """
        Initialize dataset
        
        Args:
            json_path: JSON file path
            tokenizer: tokenizer
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
            raise FileNotFoundError(f"JSON file does not exist: {json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON file should contain a list")
            
            for i, item in enumerate(tqdm(data, desc="Loading samples")):
                try:
                    # Check required fields
                    if not isinstance(item, dict):
                        print(f"Warning: Item {i} is not in dictionary format, skipping")
                        continue
                    
                    if 'label' not in item or 'input' not in item:
                        print(f"Warning: Item {i} missing 'label' or 'input' field, skipping")
                        continue
                    
                    # Get label
                    label = item['label']
                    if not isinstance(label, int):
                        try:
                            label = int(label)
                        except:
                            print(f"Warning: Label {label} in item {i} is not an integer, skipping")
                            continue
                    
                    if label not in [0, 1, 2, 3]:
                        print(f"Warning: Label {label} in item {i} is not in valid range [0,1,2,3], skipping")
                        continue
                    
                    # Get text
                    text = item['input']
                    if not text or not isinstance(text, str):
                        print(f"Warning: Text in item {i} is empty or not a string, skipping")
                        continue
                    
                    # Get image filename (if available)
                    image_name = item.get('name', f"sample_{i}")  # Modified: get from 'name' field, not 'image_name'
                    
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
        """Analyze label distribution (identical to Code A)"""
        labels_array = np.array(self.labels)
        num_samples = len(labels_array)
        
        print(f"\nDataset statistics:")
        print(f"Total samples: {num_samples}")
        
        # Label distribution
        self.activation_levels = ["Level 0 (no activation)", "Level 1 (mild activation)", "Level 2 (moderate activation)", "Level 3 (severe activation)"]
        self.class_counts = np.bincount(labels_array, minlength=4)
        
        for level in range(4):
            count = self.class_counts[level]
            ratio = count / num_samples
            print(f"  {self.activation_levels[level]}: {count} samples ({ratio:.2%})")
        
        # Calculate class imbalance ratio
        max_count = np.max(self.class_counts)
        min_count = np.min(self.class_counts[self.class_counts > 0]) if np.any(self.class_counts > 0) else 0
        
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Tokenization method identical to Code A
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
            'image_name': self.image_names[idx]  # Modified: use image name from 'name' field
        }

# --------------------- Evaluator Class ---------------------
class LangerhansEvaluator:
    def __init__(self, device: str = 'cuda'):
        """
        Initialize evaluator
        
        Args:
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"✓ Evaluator initialized")
        print(f"  Device: {self.device}")
    
    def load_model(self, model_weights_path: str, base_model_path: str, 
                   tokenizer_path: str = None, feature_dim: int = 256, 
                   num_classes: int = 4, use_slow_tokenizer: bool = True):
        """
        Load .pth weight file (consistent with Code A's loading method)
        
        Args:
            model_weights_path: Path to .pth weight file
            base_model_path: Base model path (for creating model structure)
            tokenizer_path: Tokenizer path (if different from model path)
            feature_dim: Feature dimension
            num_classes: Number of classes
            use_slow_tokenizer: Whether to use slow tokenizer
        """
        print(f"\nLoading model...")
        print(f"Weight file: {model_weights_path}")
        print(f"Base model: {base_model_path}")
        print(f"Tokenizer path: {tokenizer_path if tokenizer_path else 'Same as base model'}")
        
        # 1. Load tokenizer (same method as Code A)
        print("\n1. Loading tokenizer...")
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
        print(f"✓ Tokenizer loaded")
        print(f"  Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"  Using fast tokenizer: {not use_slow_tokenizer}")
        
        # 2. Create model structure (identical to Code A)
        print("\n2. Creating model structure...")
        self.model = T5EncoderForActivationLevel(
            base_model_path=base_model_path,
            feature_dim=feature_dim,
            num_classes=num_classes,
            use_pretrained_encoder=False,  # Pretrained encoder not needed for evaluation
            pretrained_generative_model_path=None
        )
        
        # 3. Load .pth weights (same loading method as Code A)
        print(f"\n3. Loading weight file: {model_weights_path}")
        
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Weight file does not exist: {model_weights_path}")
        
        # Load weights (using same map_location as Code A)
        state_dict = torch.load(model_weights_path, map_location="cpu")
        
        # Key point: Code A saves the trained state_dict directly, so load it directly
        # No need to check for 'state_dict' key since Code A directly saves model.state_dict()
        self.model.load_state_dict(state_dict)
        
        print(f"✓ Successfully loaded weight file")
        
        # 4. Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model moved to {self.device}, set to evaluation mode")
        
        # Print model parameter information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def evaluate(self, dataloader: DataLoader, detailed: bool = True, 
                save_predictions: bool = True, predictions_file: str = None):
        """
        Evaluate model (consistent with Code A's evaluate method)
        
        Args:
            dataloader: Data loader
            detailed: Whether to display detailed results
            save_predictions: Whether to save predictions
            predictions_file: Path to save predictions
        """
        print("\n=== Evaluating Model (Langerhans Cell Activation Level Classification) ===")
        
        all_probs = []
        all_logits = []
        all_labels = []
        all_image_names = []
        
        # Use torch.no_grad() to ensure consistency with Code A evaluation
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                image_names = batch['image_name']
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                # Collect results
                all_probs.append(probs.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_image_names.extend(image_names)
        
        # Combine results from all batches
        all_probs = np.concatenate(all_probs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Get predicted classes
        preds = np.argmax(all_probs, axis=1)
        
        # Calculate evaluation metrics (using same get_metrix function as Code A)
        wAcc_result, wSe_result, wSp_result = get_metrix(preds, all_labels)
        
        # Save predictions to specified path
        if save_predictions and predictions_file:
            self._save_predictions(all_probs, all_labels, all_image_names, preds, predictions_file)
        
        # Display detailed results
        if detailed:
            self._print_detailed_results(wAcc_result, wSe_result, wSp_result, 
                                       all_labels, preds)
        
        # Return results (same format as Code A's evaluate method)
        results = {
            'wAcc': wAcc_result,
            'wSe': wSe_result,
            'wSp': wSp_result,
            'num_samples': len(all_labels)
        }
        
        return results, all_probs, all_labels, preds, all_image_names
    
    def _save_predictions(self, all_probs, all_labels, all_image_names, preds, predictions_file):
        """Save prediction results to JSON file (modified: directly get name from dataset)"""
        print(f"\nSaving prediction results to: {predictions_file}")
        
        predictions_data = []
        for i in range(len(all_image_names)):
            predictions_data.append({
                'name': all_image_names[i],  # Name directly obtained from dataset
                'text_prediction_probs': all_probs[i].tolist(),
                'true_label': int(all_labels[i])
            })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Prediction results saved to: {predictions_file}")
        print(f"  Saved {len(predictions_data)} prediction records")
    
    def _print_detailed_results(self, wAcc_result, wSe_result, wSp_result, 
                              all_labels, preds):
        """Print detailed results (using PrettyTable format, identical to Code A)"""
        print("\n" + "=" * 70)
        print("Detailed Evaluation Report")
        print("=" * 70)
        
        print(f"\nOverall Metrics:")
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["Total Samples", len(all_labels)])
        table.add_row(["Weighted Accuracy (wAcc)", f"{wAcc_result[0]:.4f}"])
        table.add_row(["Weighted Sensitivity (wSe)", f"{wSe_result[0]:.4f}"])
        table.add_row(["Weighted Specificity (wSp)", f"{wSp_result[0]:.4f}"])
        print(table)
        
        print(f"\nDetailed Metrics by Class:")
        activation_levels = ["Level 0 (no activation)", "Level 1 (mild activation)", "Level 2 (moderate activation)", "Level 3 (severe activation)"]
        
        if len(wAcc_result) > 1 and len(wSe_result) > 1 and len(wSp_result) > 1:
            class_table = PrettyTable()
            class_table.field_names = ["Activation Level", "Accuracy", "Sensitivity", "Specificity"]
            for i in range(4):
                class_table.add_row([
                    activation_levels[i],
                    f"{wAcc_result[1][i]:.4f}",
                    f"{wSe_result[1][i]:.4f}",
                    f"{wSp_result[1][i]:.4f}"
                ])
            print(class_table)

# --------------------- Main Evaluation Function ---------------------
def evaluate_model(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    # Set random seed (same as Code A)
    set_seed(args.seed)
    
    print("=" * 80)
    print("Langerhans Cell Activation Level Classification Model Evaluation")
    print("=" * 80)
    
    # Device selection (same as Code A)
    if args.device == 'cuda':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU")
    elif args.device == 'cpu':
        device = "cpu"
    else:  # 'auto'
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    try:
        # Create evaluator
        evaluator = LangerhansEvaluator(device=device)
        
        # 1. Load model (using same parameters as Code A training)
        evaluator.load_model(
            model_weights_path=args.model_weights_path,
            base_model_path=args.base_model,
            tokenizer_path=args.tokenizer_path,
            feature_dim=args.feature_dim,
            num_classes=4,
            use_slow_tokenizer=args.use_slow_tokenizer
        )
        
        # 2. Load dataset
        print(f"\nLoading evaluation dataset: {args.eval_json}")
        dataset = LangerhansJsonDataset(
            json_path=args.eval_json,
            tokenizer=evaluator.tokenizer,
            max_len=args.max_len
        )
        
        # 3. Create data loader (same parameters as Code A evaluation)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,  # No shuffle during evaluation
            num_workers=0  # No multi-threading for reproducibility
        )
        print(f"✓ Created data loader: {len(dataset)} samples, batch size: {args.batch_size}")
        
        # 4. Evaluate model and save predictions
        results, probs, labels, preds, image_names = evaluator.evaluate(
            dataloader=dataloader,
            detailed=True,
            save_predictions=args.save_predictions,
            predictions_file=args.predictions_file
        )
        
        # 5. Print final summary
        print("\n" + "=" * 70)
        print("Evaluation Complete - Summary")
        print("=" * 70)
        print(f"  Model weights: {args.model_weights_path}")
        print(f"  Dataset: {args.eval_json}")
        print(f"  Number of samples: {len(labels)}")
        print(f"  Weighted Accuracy (wAcc): {results['wAcc'][0]:.4f}")
        print(f"  Weighted Sensitivity (wSe): {results['wSe'][0]:.4f}")
        print(f"  Weighted Specificity (wSp): {results['wSp'][0]:.4f}")
        
        if args.save_predictions:
            print(f"  Predictions saved: {args.predictions_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

# --------------------- Command Line Interface ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Langerhans Cell Activation Level Classification Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # ================ Required Parameters ================
    parser.add_argument('--model_weights_path', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/checkpoint/Finetune_model/best_model_epoch_10_wacc_0.8801.pth",
                       help='Path to model .pth weight file')
    
    parser.add_argument('--eval_json', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data/LCs_finetune_test.json",
                       help='Path to evaluation data JSON file')
    
    # ================ Output Parameters ================
    parser.add_argument('--predictions_file', type=str,
                       default="/data/Desktop/BioMiner/Generative_model/text_grading_predictions.json",
                       help='Path to save prediction results')
    
    parser.add_argument('--save_predictions', action='store_true', default=True,
                       help='Whether to save prediction results')
    
    # ================ Model Path Parameters (Same as Code A) ================
    parser.add_argument('--base_model', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/models/t5-clinical-base",
                       help='Base T5 model path')
    
    parser.add_argument('--tokenizer_path', type=str, 
                       default="/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer/",
                       help='Tokenizer path (if different from model path)')
    
    # ================ Evaluation Parameters (Same as Code A evaluation) ================
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension')
    
    parser.add_argument('--max_len', type=int, default=512,
                       help='Maximum sequence length')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--use_slow_tokenizer', action='store_true', default=False,
                       help='Whether to use slow tokenizer')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Check necessary paths
    if not os.path.exists(args.model_weights_path):
        print(f"❌ Error: Model weight file does not exist: {args.model_weights_path}")
        exit(1)
    
    if not os.path.exists(args.eval_json):
        print(f"❌ Error: Evaluation data path does not exist: {args.eval_json}")
        exit(1)
    
    if not os.path.exists(args.base_model):
        print(f"❌ Error: Base model path does not exist: {args.base_model}")
        exit(1)
    
    # Check tokenizer path
    if args.tokenizer_path and not os.path.exists(args.tokenizer_path):
        print(f"⚠ Warning: Specified tokenizer path does not exist: {args.tokenizer_path}")
        print("  Will use base model path as tokenizer path")
        args.tokenizer_path = None
    
    print(f"\nConfiguration Check (ensure consistency with Code A training configuration):")
    print(f"  Model weights: {args.model_weights_path}")
    print(f"  Evaluation data: {args.eval_json}")
    print(f"  Prediction save path: {args.predictions_file}")
    print(f"  Save predictions: {args.save_predictions}")
    print(f"  Base model: {args.base_model}")
    print(f"  Tokenizer path: {args.tokenizer_path if args.tokenizer_path else args.base_model}")
    print(f"  Feature dimension: {args.feature_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Maximum length: {args.max_len}")
    print(f"  Random seed: {args.seed}")
    print(f"  Slow tokenizer: {args.use_slow_tokenizer}")
    
    # Run evaluation
    evaluate_model(args)
