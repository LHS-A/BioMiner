# This script evaluates the Stage1_generative_pretraining.py code; simply change the --model_path argument at the bottom and run to obtain Rouge-1, Rouge-2, and Rouge-L scores.
import sys
import os
import json
import argparse
import datetime
from pathlib import Path

# Get the parent directory of the current file (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the project root to Python path so other project modules can be imported
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

# Import project-specific modules
try:
    from Generative_model.utils.nltoolkit import init_nltk, postprocess_text
except ImportError as e:
    print(f"Warning: nltoolkit could not be imported: {e}")
    # Fallback functions
    def init_nltk():
        print("Initializing NLTK (fallback)")
    
    def postprocess_text(preds, labels):
        """Fallback text post-processing"""
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

import evaluate

def evaluate_generation_model(model_path, config=None):
    """
    Evaluate generative model performance
    
    Args:
        model_path: path to model weights (supports only best_model and final_model formats)
        config: configuration object; if None, default settings are used
    """
    
    # Initialize NLTK
    init_nltk()
    
    # Create config object if none provided
    if config is None:
        # Use the actual parameters you provided
        class Config:
            def __init__(self):
                self.dataset_name = "LCs_grading"
                self.dataset_path = "/data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data"
                self.history_column = "observation"
                self.future_column = "forecast"
                self.max_source_length = 1024  # per your config
                self.max_target_length = 128   # per your config
                self.per_device_eval_batch_size = 16  # per your config
                self.preprocessing_num_workers = 0    # per your config
                self.overwrite_cache = True
                self.pad_to_max_length = False
                self.use_slow_tokenizer = False       # per your config
                self.num_beams = 1
                self.cache_dir = "./.cache/"
                self.tokenizer_name = "/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer/"
        
        config = Config()
    
    # Parse model path, allowing only best_model and final_model
    model_path = Path(model_path)
    print(f"\n{'='*60}")
    print(f"🔍 Model Evaluation Config")
    print(f"{'='*60}")
    
    # Check model type, allowing only best_model and final_model
    if model_path.name == "best_model":
        model_type = "best_model"
        # Try to load best-model info
        best_info_path = model_path / "best_model_info.json"
        if best_info_path.exists():
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
                print(f"📊 Best-model info:")
                print(f"   - Best epoch: {best_info.get('best_epoch', 'N/A')}")
                print(f"   - Best ROUGE-1 score: {best_info.get('best_metric', 'N/A'):.2f}%")
        else:
            print("ℹ️ Best-model info file not found")
    elif model_path.name == "final_model":
        model_type = "final_model"
        print(f"📁 Loading final model")
    else:
        # Epoch models are not allowed
        if model_path.name.startswith("epoch_"):
            raise ValueError(f"Epoch-model evaluation not supported: {model_path.name}; only best_model and final_model are allowed")
        else:
            model_type = "other"
            print(f"⚠️ Warning: evaluating non-standard model: {model_path.name}")
    
    print(f"🔍 Model path: {model_path}")
    print(f"📊 Model type: {model_type}")
    
    # Build pre-processed validation dataset path
    preprocessed_val_dataset_name = f"{config.dataset_name}_val.json"
    data_file = os.path.join(config.dataset_path, preprocessed_val_dataset_name)
    
    # Check dataset file existence
    if not os.path.exists(data_file):
        # Try alternative paths
        alt_path = os.path.join(config.dataset_path, "..", preprocessed_val_dataset_name)
        if os.path.exists(alt_path):
            data_file = alt_path
            print(f"ℹ️ Dataset found at alternative path: {alt_path}")
        else:
            raise FileNotFoundError(f"Validation file not found: {data_file}")
    
    print(f"📂 Validation dataset: {data_file}")
    
    # Load validation set
    extension = data_file.split(".")[-1]
    try:
        raw_datasets = load_dataset(extension, data_files={"validation": data_file}, cache_dir=config.cache_dir)
        val_dataset = raw_datasets["validation"]
        print(f"✅ Validation set loaded, samples: {len(val_dataset)}")
    except Exception as e:
        print(f"❌ Validation set loading failed: {e}")
        raise
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer...")
    tokenizer_path = config.tokenizer_name
    
    # Check tokenizer path existence
    if not os.path.exists(tokenizer_path):
        print(f"⚠️ Tokenizer path not found: {tokenizer_path}")
        # Try loading from model path
        if (model_path / "tokenizer.json").exists() or (model_path / "vocab.json").exists() or (model_path / "special_tokens_map.json").exists():
            tokenizer_path = model_path
            print(f"ℹ️ Loading tokenizer from model path: {tokenizer_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=False,
            cache_dir=config.cache_dir,
            use_fast=not config.use_slow_tokenizer
        )
        print(f"✅ Tokenizer loaded")
        print(f"   - Vocab size: {len(tokenizer)}")
        print(f"   - Pad token: {tokenizer.pad_token}")
        print(f"   - Using fast tokenizer: {not config.use_slow_tokenizer}")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        # Fallback to default T5 tokenizer
        print("Trying T5 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",
            trust_remote_code=False,
            cache_dir=config.cache_dir,
            use_fast=True
        )
    
    # Load model
    print(f"\n🤖 Loading generation model...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            trust_remote_code=False,
            cache_dir=config.cache_dir
        )
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✅ Model loaded")
        print(f"   - Device: {device}")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Architecture: {model.__class__.__name__}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        # Try loading from possible parent directory
        parent_path = model_path.parent
        possible_paths = list(parent_path.glob("*pytorch_model.bin")) + list(parent_path.glob("*model*.bin"))
        if possible_paths:
            print(f"Trying to load from parent directory: {parent_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(parent_path),
                trust_remote_code=False,
                cache_dir=config.cache_dir
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            print(f"✅ Loaded successfully from parent directory")
        else:
            raise
    
    # Data preprocessing function
    def preprocess_function(examples):
        """
        Pre-process function converting text data to model input format
        """
        inputs = examples[config.history_column]       # input data (historical trajectory)
        targets = examples[config.future_column]       # target data (future trajectory)

        # Tokenize inputs
        padding = "max_length" if config.pad_to_max_length else False
        model_inputs = tokenizer(inputs, max_length=config.max_source_length, padding=padding, truncation=True)

        # Tokenize targets
        labels = tokenizer(text_target=targets, max_length=config.max_target_length, padding=padding, truncation=True)

        # Handle padding tokens; -100 tells the model to ignore these positions in loss computation
        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in
                                   labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Pre-process validation set
    print(f"\n🔧 Pre-processing validation data...")
    print(f"   - Max input length: {config.max_source_length}")
    print(f"   - Max output length: {config.max_target_length}")
    print(f"   - Using padding: {config.pad_to_max_length}")
    
    column_names = val_dataset.column_names
    print(f"   - Dataset columns: {column_names}")
    
    # Check required columns exist
    if config.history_column not in column_names:
        print(f"❌ Column '{config.history_column}' not found in dataset")
        print(f"   Available columns: {column_names}")
        raise ValueError(f"Column '{config.history_column}' not found")
    
    if config.future_column not in column_names:
        print(f"❌ Column '{config.future_column}' not found in dataset")
        print(f"   Available columns: {column_names}")
        raise ValueError(f"Column '{config.future_column}' not found")
    
    with torch.no_grad():
        processed_val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Pre-processing validation set"
        )
    
    print(f"✅ Data pre-processing complete, samples processed: {len(processed_val_dataset)}")
    
    # Create data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    # Create data loader
    eval_dataloader = DataLoader(
        processed_val_dataset, 
        collate_fn=data_collator, 
        batch_size=config.per_device_eval_batch_size,
        shuffle=False
    )
    
    print(f"📊 Data loader created, batch size: {config.per_device_eval_batch_size}")
    print(f"   Total batches: {len(eval_dataloader)}")
    
    # Load ROUGE metric (excluding RougeLSum)
    print(f"\n📈 Loading evaluation metrics...")
    local_rouge_path = "/data/Desktop/BioMiner/Generative_model/rouge/rouge.py"
    try:
        metric = evaluate.load(local_rouge_path)
        print(f"✅ ROUGE evaluator loaded")
        print(f"   Using local ROUGE evaluator: {local_rouge_path}")
    except Exception as e:
        print(f"⚠️ Could not load local ROUGE evaluator: {e}")
        print("Falling back to Hugging Face ROUGE evaluator...")
        metric = evaluate.load("rouge")
    
    # Evaluate model
    print(f"\n{'='*60}")
    print(f"🚀 Starting generative model evaluation")
    print(f"{'='*60}")
    
    all_predictions = []
    all_references = []
    all_inputs = []
    example_shown = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluation progress")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate predictions
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=config.max_target_length,
                min_length=10,
                length_penalty=2.0,
                num_beams=config.num_beams
            )
            
            # Decode generated tokens and labels
            labels = batch["labels"]
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            
            # Display one generation example
            if not example_shown and batch_idx == 0:
                print(f"\n{'📝 Generation Example':-^40}")
                input_text = decoded_inputs[0]
                # Limit display length
                display_len = min(200, len(input_text))
                print(f"Input ({len(input_text)} chars):")
                print(f"{input_text[:display_len]}{'...' if len(input_text) > display_len else ''}")
                print(f"\nGround truth: {decoded_labels[0]}")
                print(f"Model prediction: {decoded_preds[0]}")
                print(f"{'-'*40}")
                example_shown = True
            
            # Post-process text
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
            # Collect all predictions and references
            all_predictions.extend(decoded_preds)
            all_references.extend(decoded_labels)
            all_inputs.extend(decoded_inputs)
            
            # Add batch to metric computation
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    # Compute final metrics
    print(f"\n📊 Computing evaluation metrics...")
    result = metric.compute(use_stemmer=True)
    
    # Filter out RougeLSum (per your request)
    filtered_result = {}
    for key, value in result.items():
        if key != "rougeLsum":
            filtered_result[key] = round(value * 100, 4)
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"🎯 Evaluation Results")
    print(f"{'='*60}")
    
    # Group metrics by type
    rouge_metrics = {k: v for k, v in filtered_result.items() if k.startswith("rouge")}
    other_metrics = {k: v for k, v in filtered_result.items() if not k.startswith("rouge")}
    
    if rouge_metrics:
        print(f"\n📈 ROUGE Metrics:")
        for metric_name, score in sorted(rouge_metrics.items()):
            print(f"  {metric_name:12s}: {score:6.2f}%")
    
    if other_metrics:
        print(f"\n📊 Other Metrics:")
        for metric_name, score in sorted(other_metrics.items()):
            print(f"  {metric_name:12s}: {score:6.2f}%")
    
    # Compute average ROUGE (excluding RougeLSum)
    rouge_scores = [v for k, v in rouge_metrics.items() if k != "rougeLsum"]
    if rouge_scores:
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        print(f"\n📊 Average ROUGE (excluding RougeLSum): {avg_rouge:.2f}%")
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"📋 Evaluation Summary")
    print(f"{'='*60}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Sample count: {len(all_predictions)}")
    print(f"Batch size: {config.per_device_eval_batch_size}")
    print(f"Max input length: {config.max_source_length}")
    print(f"Max output length: {config.max_target_length}")
    print(f"\nKey metrics:")
    print(f"  - ROUGE-1: {filtered_result.get('rouge1', 0):.2f}%")
    print(f"  - ROUGE-2: {filtered_result.get('rouge2', 0):.2f}%")
    print(f"  - ROUGE-L: {filtered_result.get('rougeL', 0):.2f}%")
    
    # If best model, show comparison
    if model_type == "best_model" and 'best_info' in locals():
        original_score = best_info.get("best_metric", 0)
        current_score = filtered_result.get('rouge1', 0)
        print(f"\n📊 Comparison with training-time best:")
        print(f"  - Training best ROUGE-1: {original_score:.2f}%")
        print(f"  - Current eval ROUGE-1: {current_score:.2f}%")
        difference = current_score - original_score
        if difference > 0:
            print(f"  - Change: +{difference:.2f}% (better)")
        elif difference < 0:
            print(f"  - Change: {difference:.2f}% (worse)")
        else:
            print(f"  - Change: 0.00% (same)")
    
    # Model quality assessment
    rouge1_score = filtered_result.get('rouge1', 0)
    if rouge1_score >= 40:
        quality = "Excellent ★★★★☆"
        quality_desc = "Very high generation quality, near human level"
    elif rouge1_score >= 30:
        quality = "Good ★★★☆☆"
        quality_desc = "Good generation quality, usable"
    elif rouge1_score >= 20:
        quality = "Fair ★★☆☆☆"
        quality_desc = "Average generation quality, needs further optimization"
    elif rouge1_score >= 10:
        quality = "Needs improvement ★☆☆☆☆"
        quality_desc = "Low generation quality, significant improvement required"
    else:
        quality = "Poor ☆☆☆☆☆"
        quality_desc = "Very poor generation quality, retraining recommended"
    
    print(f"\nModel quality assessment:")
    print(f"  - Rating: {quality}")
    print(f"  - Description: {quality_desc}")
    print(f"  - Based on ROUGE-1 score: {rouge1_score:.2f}%")
    
    print(f"\n💡 Tip: Evaluation complete; results are console-only, no JSON file generated")
    print(f"{'='*60}")
    
    return filtered_result

def main():
    """Main: parse CLI arguments and run evaluation"""
    parser = argparse.ArgumentParser(
        description="Evaluate generative model (supports only best_model and final_model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/data/Desktop/BioMiner/Generative_model/checkpoint/Generative_model/final_model",
        help="Model weights path (supports only best_model and final_model formats)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="LCs_grading",
        help="Dataset name, default LCs_grading"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data",
        help="Dataset path"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Evaluation batch size, default 16"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer/",
        help="Tokenizer path"
    )
    parser.add_argument(
        "--max_source_length", 
        type=int, 
        default=1024,
        help="Max input length, default 1024"
    )
    parser.add_argument(
        "--max_target_length", 
        type=int, 
        default=128,
        help="Max output length, default 128"
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=1,
        help="Beam search size, default 1"
    )
    
    args = parser.parse_args()
    
    # Print argument info
    print(f"\n{'='*60}")
    print(f"🤖 Generative Model Evaluation Tool")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Model path: {args.model_path}")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Dataset path: {args.dataset_path}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max input length: {args.max_source_length}")
    print(f"  - Max output length: {args.max_target_length}")
    print(f"  - Beam search size: {args.num_beams}")
    print(f"{'='*60}")
    
    # Create config object
    class Config:
        def __init__(self, args):
            self.dataset_name = args.dataset
            self.dataset_path = args.dataset_path
            self.history_column = "observation"
            self.future_column = "forecast"
            self.max_source_length = args.max_source_length
            self.max_target_length = args.max_target_length
            self.per_device_eval_batch_size = args.batch_size
            self.preprocessing_num_workers = 0  # per your config
            self.overwrite_cache = True
            self.pad_to_max_length = False
            self.use_slow_tokenizer = False     # per your config
            self.num_beams = args.num_beams
            self.cache_dir = "./.cache/"
            self.tokenizer_name = args.tokenizer_path
    
    config = Config(args)
    
    # Run evaluation
    try:
        results = evaluate_generation_model(args.model_path, config)
        print(f"\n✅ Evaluation complete!")
        return results
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
