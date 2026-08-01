# Compute and report evaluation metrics.
import sys
import os
import json
import argparse
import datetime
from pathlib import Path

# Ensure the output directory exists.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Resolve the required path.
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

# Import the required project component.
try:
    from Generative_model.utils.nltoolkit import init_nltk, postprocess_text
except ImportError as e:
    print(f"警告: 无法导入nltoolkit: {e}")
    # Retain this implementation detail from the original pipeline.
    def init_nltk():
        print("初始化NLTK（备用）")
    
    def postprocess_text(preds, labels):
        """Documentation for this retained evaluation component."""
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

import evaluate

def evaluate_generation_model(model_path, config=None):
    """Documentation for this retained evaluation component."""
    
    # Retain this implementation detail from the original pipeline.
    init_nltk()
    
    # Create the required object.
    if config is None:
        # Retain this implementation detail from the original pipeline.
        class Config:
            def __init__(self):
                self.dataset_name = "LCs_grading"
                self.dataset_path = "/data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data"
                self.history_column = "observation"
                self.future_column = "forecast"
                self.max_source_length = 1024  # Retain this implementation detail from the original pipeline.
                self.max_target_length = 128   # Retain this implementation detail from the original pipeline.
                self.per_device_eval_batch_size = 16  # Retain this implementation detail from the original pipeline.
                self.preprocessing_num_workers = 0    # Retain this implementation detail from the original pipeline.
                self.overwrite_cache = True
                self.pad_to_max_length = False
                self.use_slow_tokenizer = False       # Retain this implementation detail from the original pipeline.
                self.num_beams = 1
                self.cache_dir = "./.cache/"
                self.tokenizer_name = "/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer/"
        
        config = Config()
    
    # Configure or use the model.
    model_path = Path(model_path)
    print(f"\n{'='*60}")
    print(f"🔍 模型评估配置")
    print(f"{'='*60}")
    
    # Configure or use the model.
    if model_path.name == "best_model":
        model_type = "best_model"
        # Configure or use the model.
        best_info_path = model_path / "best_model_info.json"
        if best_info_path.exists():
            with open(best_info_path, 'r') as f:
                best_info = json.load(f)
                print(f"📊 最佳模型信息:")
                print(f"   - 最佳epoch: {best_info.get('best_epoch', 'N/A')}")
                print(f"   - 最佳ROUGE-1分数: {best_info.get('best_metric', 'N/A'):.2f}%")
        else:
            print("ℹ️ 未找到最佳模型信息文件")
    elif model_path.name == "final_model":
        model_type = "final_model"
        print(f"📁 加载最终模型")
    else:
        # Compute and report evaluation metrics.
        if model_path.name.startswith("epoch_"):
            raise ValueError(f"不支持评估epoch模型: {model_path.name}，只支持评估best_model和final_model")
        else:
            model_type = "other"
            print(f"⚠️ 警告: 评估非标准模型: {model_path.name}")
    
    print(f"🔍 模型路径: {model_path}")
    print(f"📊 模型类型: {model_type}")
    
    # Load or process the dataset.
    preprocessed_val_dataset_name = f"{config.dataset_name}_val.json"
    data_file = os.path.join(config.dataset_path, preprocessed_val_dataset_name)
    
    # Load or process the dataset.
    if not os.path.exists(data_file):
        # Resolve the required path.
        alt_path = os.path.join(config.dataset_path, "..", preprocessed_val_dataset_name)
        if os.path.exists(alt_path):
            data_file = alt_path
            print(f"ℹ️ 在备用路径找到数据集: {alt_path}")
        else:
            raise FileNotFoundError(f"验证集文件不存在: {data_file}")
    
    print(f"📂 验证数据集: {data_file}")
    
    # Load the required artifact.
    extension = data_file.split(".")[-1]
    try:
        raw_datasets = load_dataset(extension, data_files={"validation": data_file}, cache_dir=config.cache_dir)
        val_dataset = raw_datasets["validation"]
        print(f"✅ 验证集加载成功，样本数: {len(val_dataset)}")
    except Exception as e:
        print(f"❌ 验证集加载失败: {e}")
        raise
    
    # Load the required artifact.
    print(f"\n🔤 加载分词器...")
    tokenizer_path = config.tokenizer_name
    
    # Resolve the required path.
    if not os.path.exists(tokenizer_path):
        print(f"⚠️ 分词器路径不存在: {tokenizer_path}")
        # Configure or use the model.
        if (model_path / "tokenizer.json").exists() or (model_path / "vocab.json").exists() or (model_path / "special_tokens_map.json").exists():
            tokenizer_path = model_path
            print(f"ℹ️ 从模型路径加载分词器: {tokenizer_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=False,
            cache_dir=config.cache_dir,
            use_fast=not config.use_slow_tokenizer
        )
        print(f"✅ 分词器加载成功")
        print(f"   - 词汇表大小: {len(tokenizer)}")
        print(f"   - 填充token: {tokenizer.pad_token}")
        print(f"   - 使用fast tokenizer: {not config.use_slow_tokenizer}")
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        # Retain this implementation detail from the original pipeline.
        print("尝试使用T5 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",
            trust_remote_code=False,
            cache_dir=config.cache_dir,
            use_fast=True
        )
    
    # Configure or use the model.
    print(f"\n🤖 加载生成模型...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            trust_remote_code=False,
            cache_dir=config.cache_dir
        )
        # Configure or use the model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✅ 模型加载成功")
        print(f"   - 使用设备: {device}")
        print(f"   - 参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - 模型架构: {model.__class__.__name__}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        # Ensure the output directory exists.
        parent_path = model_path.parent
        possible_paths = list(parent_path.glob("*pytorch_model.bin")) + list(parent_path.glob("*model*.bin"))
        if possible_paths:
            print(f"尝试从父目录加载: {parent_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(parent_path),
                trust_remote_code=False,
                cache_dir=config.cache_dir
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            print(f"✅ 从父目录加载成功")
        else:
            raise
    
    # Load or process the dataset.
    def preprocess_function(examples):
        """Documentation for this retained evaluation component."""
        inputs = examples[config.history_column]       # Load or process the dataset.
        targets = examples[config.future_column]       # Load or process the dataset.

        # Prepare the model input.
        padding = "max_length" if config.pad_to_max_length else False
        model_inputs = tokenizer(inputs, max_length=config.max_source_length, padding=padding, truncation=True)

        # Retain this implementation detail from the original pipeline.
        labels = tokenizer(text_target=targets, max_length=config.max_target_length, padding=padding, truncation=True)

        # Configure or use the model.
        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in
                                   labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Process the current value.
    print(f"\n🔧 预处理验证数据...")
    print(f"   - 最大输入长度: {config.max_source_length}")
    print(f"   - 最大输出长度: {config.max_target_length}")
    print(f"   - 使用padding: {config.pad_to_max_length}")
    
    column_names = val_dataset.column_names
    print(f"   - 数据集列名: {column_names}")
    
    # Validate the required condition.
    if config.history_column not in column_names:
        print(f"❌ 列 '{config.history_column}' 不存在于数据集中")
        print(f"   可用列: {column_names}")
        raise ValueError(f"列 '{config.history_column}' 不存在")
    
    if config.future_column not in column_names:
        print(f"❌ 列 '{config.future_column}' 不存在于数据集中")
        print(f"   可用列: {column_names}")
        raise ValueError(f"列 '{config.future_column}' 不存在")
    
    with torch.no_grad():
        processed_val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="预处理验证集"
        )
    
    print(f"✅ 数据预处理完成，处理样本数: {len(processed_val_dataset)}")
    
    # Load or process the dataset.
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    # Load or process the dataset.
    eval_dataloader = DataLoader(
        processed_val_dataset, 
        collate_fn=data_collator, 
        batch_size=config.per_device_eval_batch_size,
        shuffle=False
    )
    
    print(f"📊 创建数据加载器，批次大小: {config.per_device_eval_batch_size}")
    print(f"   总批次数: {len(eval_dataloader)}")
    
    # Compute and report evaluation metrics.
    print(f"\n📈 加载评估指标...")
    local_rouge_path = "/data/Desktop/BioMiner/Generative_model/rouge/rouge.py"
    try:
        metric = evaluate.load(local_rouge_path)
        print(f"✅ ROUGE评估器加载成功")
        print(f"   使用本地ROUGE评估器: {local_rouge_path}")
    except Exception as e:
        print(f"⚠️ 无法加载本地ROUGE评估器: {e}")
        print("使用Hugging Face的ROUGE评估器...")
        metric = evaluate.load("rouge")
    
    # Compute and report evaluation metrics.
    print(f"\n{'='*60}")
    print(f"🚀 开始评估生成模型性能")
    print(f"{'='*60}")
    
    all_predictions = []
    all_references = []
    all_inputs = []
    example_shown = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="评估进度")):
            # Load or process the dataset.
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate or collect predictions.
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=config.max_target_length,
                min_length=10,
                length_penalty=2.0,
                num_beams=config.num_beams
            )
            
            # Process the grading labels.
            labels = batch["labels"]
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            
            # Retain this implementation detail from the original pipeline.
            if not example_shown and batch_idx == 0:
                print(f"\n{'📝 生成示例':-^40}")
                input_text = decoded_inputs[0]
                # Retain this implementation detail from the original pipeline.
                display_len = min(200, len(input_text))
                print(f"输入 ({len(input_text)} 字符):")
                print(f"{input_text[:display_len]}{'...' if len(input_text) > display_len else ''}")
                print(f"\n真实输出: {decoded_labels[0]}")
                print(f"模型预测: {decoded_preds[0]}")
                print(f"{'-'*40}")
                example_shown = True
            
            # Process the current value.
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
            # Generate or collect predictions.
            all_predictions.extend(decoded_preds)
            all_references.extend(decoded_labels)
            all_inputs.extend(decoded_inputs)
            
            # Compute the required value.
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    # Compute and report evaluation metrics.
    print(f"\n📊 计算评估指标...")
    result = metric.compute(use_stemmer=True)
    
    # Retain this implementation detail from the original pipeline.
    filtered_result = {}
    for key, value in result.items():
        if key != "rougeLsum":
            filtered_result[key] = round(value * 100, 4)
    
    # Report the current status.
    print(f"\n{'='*60}")
    print(f"🎯 评估结果")
    print(f"{'='*60}")
    
    # Retain this implementation detail from the original pipeline.
    rouge_metrics = {k: v for k, v in filtered_result.items() if k.startswith("rouge")}
    other_metrics = {k: v for k, v in filtered_result.items() if not k.startswith("rouge")}
    
    if rouge_metrics:
        print(f"\n📈 ROUGE指标:")
        for metric_name, score in sorted(rouge_metrics.items()):
            print(f"  {metric_name:12s}: {score:6.2f}%")
    
    if other_metrics:
        print(f"\n📊 其他指标:")
        for metric_name, score in sorted(other_metrics.items()):
            print(f"  {metric_name:12s}: {score:6.2f}%")
    
    # Compute the required value.
    rouge_scores = [v for k, v in rouge_metrics.items() if k != "rougeLsum"]
    if rouge_scores:
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        print(f"\n📊 平均ROUGE分数 (排除RougeLSum): {avg_rouge:.2f}%")
    
    # Retain this implementation detail from the original pipeline.
    print(f"\n{'='*60}")
    print(f"📋 评估总结")
    print(f"{'='*60}")
    print(f"模型类型: {model_type}")
    print(f"数据集: {config.dataset_name}")
    print(f"样本数量: {len(all_predictions)}")
    print(f"批次大小: {config.per_device_eval_batch_size}")
    print(f"输入最大长度: {config.max_source_length}")
    print(f"输出最大长度: {config.max_target_length}")
    print(f"\n关键指标:")
    print(f"  - ROUGE-1: {filtered_result.get('rouge1', 0):.2f}%")
    print(f"  - ROUGE-2: {filtered_result.get('rouge2', 0):.2f}%")
    print(f"  - ROUGE-L: {filtered_result.get('rougeL', 0):.2f}%")
    
    # Configure or use the model.
    if model_type == "best_model" and 'best_info' in locals():
        original_score = best_info.get("best_metric", 0)
        current_score = filtered_result.get('rouge1', 0)
        print(f"\n📊 与训练时最佳分数比较:")
        print(f"  - 训练时最佳ROUGE-1: {original_score:.2f}%")
        print(f"  - 当前评估ROUGE-1: {current_score:.2f}%")
        difference = current_score - original_score
        if difference > 0:
            print(f"  - 变化: +{difference:.2f}% (更好)")
        elif difference < 0:
            print(f"  - 变化: {difference:.2f}% (稍差)")
        else:
            print(f"  - 变化: 0.00% (相同)")
    
    # Compute and report evaluation metrics.
    rouge1_score = filtered_result.get('rouge1', 0)
    if rouge1_score >= 40:
        quality = "优秀 ★★★★☆"
        quality_desc = "模型生成质量很高，接近人类水平"
    elif rouge1_score >= 30:
        quality = "良好 ★★★☆☆"
        quality_desc = "模型生成质量良好，可以实用"
    elif rouge1_score >= 20:
        quality = "一般 ★★☆☆☆"
        quality_desc = "模型生成质量一般，需要进一步优化"
    elif rouge1_score >= 10:
        quality = "需要改进 ★☆☆☆☆"
        quality_desc = "模型生成质量较低，需要显著改进"
    else:
        quality = "较差 ☆☆☆☆☆"
        quality_desc = "模型生成质量很差，需要重新训练"
    
    print(f"\n模型质量评估:")
    print(f"  - 评级: {quality}")
    print(f"  - 描述: {quality_desc}")
    print(f"  - 基于ROUGE-1分数: {rouge1_score:.2f}%")
    
    print(f"\n💡 提示: 评估完成，结果仅显示在控制台，未生成JSON文件")
    print(f"{'='*60}")
    
    return filtered_result

def main():
    """Documentation for this retained evaluation component."""
    parser = argparse.ArgumentParser(
        description="评估生成模型性能（只支持best_model和final_model）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/data/Desktop/BioMiner/Generative_model/checkpoint/Generative_model/final_model",
        help="模型权重路径（只支持best_model和final_model格式）"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="LCs_grading",
        help="数据集名称，默认为LCs_grading"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/data/Desktop/BioMiner/Generative_model/datasets/LCs_corpus/Train_data",
        help="数据集路径"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="评估批次大小，默认为16"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer/Generative_model_tokenizer/",
        help="分词器路径"
    )
    parser.add_argument(
        "--max_source_length", 
        type=int, 
        default=1024,
        help="最大输入长度，默认为1024"
    )
    parser.add_argument(
        "--max_target_length", 
        type=int, 
        default=128,
        help="最大输出长度，默认为128"
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=1,
        help="beam search的数量，默认为1"
    )
    
    args = parser.parse_args()
    
    # Report the current status.
    print(f"\n{'='*60}")
    print(f"🤖 生成模型评估工具")
    print(f"{'='*60}")
    print(f"参数配置:")
    print(f"  - 模型路径: {args.model_path}")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 数据集路径: {args.dataset_path}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 最大输入长度: {args.max_source_length}")
    print(f"  - 最大输出长度: {args.max_target_length}")
    print(f"  - Beam search数量: {args.num_beams}")
    print(f"{'='*60}")
    
    # Create the required object.
    class Config:
        def __init__(self, args):
            self.dataset_name = args.dataset
            self.dataset_path = args.dataset_path
            self.history_column = "observation"
            self.future_column = "forecast"
            self.max_source_length = args.max_source_length
            self.max_target_length = args.max_target_length
            self.per_device_eval_batch_size = args.batch_size
            self.preprocessing_num_workers = 0  # Retain this implementation detail from the original pipeline.
            self.overwrite_cache = True
            self.pad_to_max_length = False
            self.use_slow_tokenizer = False     # Retain this implementation detail from the original pipeline.
            self.num_beams = args.num_beams
            self.cache_dir = "./.cache/"
            self.tokenizer_name = args.tokenizer_path
    
    config = Config(args)
    
    # Compute and report evaluation metrics.
    try:
        results = evaluate_generation_model(args.model_path, config)
        print(f"\n✅ 评估完成！")
        return results
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()