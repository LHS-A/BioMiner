# Prepare or inspect the dataset.
# Configure or use the model.
# Retain this implementation detail from the original training pipeline.
# Run the training step.
# Retain this implementation detail from the original training pipeline.
# Resolve the required path.

import sys
import os

# Ensure the output directory exists.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Resolve the required path.
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the required project component.
import json
import logging
import math
import os
import random

import datasets
import evaluate
import numpy as np
import torch

# Run the training step.
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, \
    get_scheduler

# Import the required project component.
from Generative_model.utils.nltoolkit import init_nltk, postprocess_text

# Retrieve the required value.
logger = get_logger(__name__)


def trainval(cfg):
    """Documentation for this retained training component."""

    # Report the current status.
    print("=== 配置参数调试 ===")
    print(f"checkpoint_path 类型: {type(cfg.checkpoint_path)}, 值: {cfg.checkpoint_path}")
    print(f"checkpoint_name 类型: {type(cfg.checkpoint_name)}, 值: {cfg.checkpoint_name}")
    print(f"dataset_name 类型: {type(cfg.dataset_name)}, 值: {cfg.dataset_name}")
    print("==================")

    # Load or validate the configuration.
    if cfg.checkpoint_name is None:
        cfg.checkpoint_name = "default_model"
        print("警告: checkpoint_name 为 None，已设置为 default_model")

    if cfg.checkpoint_path is None:
        cfg.checkpoint_path = "./checkpoint/"
        print("警告: checkpoint_path 为 None，已设置为 ./checkpoint/")

    # Retain this implementation detail from the original training pipeline.
    init_nltk()

    # Run the training step.
    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    accelerator_log_kwargs = {}
    if cfg.use_logger:
        accelerator_log_kwargs["log_with"] = cfg.logger_type
        accelerator_log_kwargs["project_dir"] = checkpoint_path

    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Load or validate the configuration.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the random seed for reproducibility.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the training checkpoint.
    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
    accelerator.wait_for_everyone()

    # Prepare or inspect the dataset.
    preprocessed_train_dataset_name = f"{cfg.dataset_name}_train.json"
    preprocessed_val_dataset_name = f"{cfg.dataset_name}_val.json"
    preprocessed_dataset_path = os.path.join(cfg.dataset_path)

    data_files = {}
    data_files["train"] = os.path.join(preprocessed_dataset_path, preprocessed_train_dataset_name)
    data_files["validation"] = os.path.join(preprocessed_dataset_path, preprocessed_val_dataset_name)

    # Prepare or inspect the dataset.
    if not os.path.exists(data_files["train"]) or not os.path.exists(data_files["validation"]):
        raise ValueError(
            f"Preprocessed dataset files not found: {data_files['train']} or {data_files['validation']}. Please run `./script/preprocessor.sh` first.")

    # Prepare or inspect the dataset.
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    # Configure or use the model.
    if cfg.model_config_name or cfg.model_name_or_path:
        config = AutoConfig.from_pretrained(cfg.model_config_name if cfg.model_config_name else cfg.model_name_or_path,
                                            trust_remote_code=False, cache_dir=cfg.cache_dir)
    else:
        config = CONFIG_MAPPING[cfg.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Configure or apply the tokenizer.
    if cfg.tokenizer_name or cfg.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path,
                                                  trust_remote_code=False, cache_dir=cfg.cache_dir,
                                                  use_fast=not cfg.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, utils/tokenizer, save it, and load it from here, using --tokenizer_name.")

    # Configure or use the model.
    if cfg.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path, config=config, trust_remote_code=False,
                                                      cache_dir=cfg.cache_dir)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=False)

    # Configure or apply the tokenizer.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    if cfg.tokenizer_name is not None:
        model.resize_token_embeddings(len(tokenizer))

    # Prepare or inspect the dataset.
    column_names = raw_datasets["train"].column_names

    history_column = cfg.history_column
    if history_column not in column_names:
        raise ValueError(
            f"--history_column' value '{cfg.history_column}' needs to be one of: {', '.join(column_names)}")
    future_column = cfg.future_column
    if future_column not in column_names:
        raise ValueError(f"--future_column' value '{cfg.future_column}' needs to be one of: {', '.join(column_names)}")

    # Retain this implementation detail from the original training pipeline.
    padding = "max_length" if cfg.pad_to_max_length else False

    # Process the current value.
    def preprocess_function(examples):
        """Documentation for this retained training component."""
        inputs = examples[history_column]       # Prepare the model input.
        targets = examples[future_column]       # Retrieve the required value.

        # Prepare the model input.
        model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, padding=padding, truncation=True)

        # Retain this implementation detail from the original training pipeline.
        labels = tokenizer(text_target=targets, max_length=cfg.max_target_length, padding=padding, truncation=True)

        # Configure or use the model.
        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in
                                   labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Prepare or inspect the dataset.
    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(preprocess_function,
                                                  batched=True,
                                                  num_proc=cfg.preprocessing_num_workers,
                                                  remove_columns=column_names,
                                                  load_from_cache_file=not cfg.overwrite_cache,
                                                  desc="Running tokenizer on train dataset")

        val_dataset = raw_datasets["validation"].map(preprocess_function,
                                                     batched=True,
                                                     num_proc=cfg.preprocessing_num_workers,
                                                     remove_columns=column_names,
                                                     load_from_cache_file=not cfg.overwrite_cache,
                                                     desc="Running tokenizer on val dataset")

    # Run the training step.
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Create the required object.
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           model=model,
                                           label_pad_token_id=label_pad_token_id,
                                           pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None)

    # Create or use the data loader.
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                                  batch_size=cfg.per_device_train_batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=cfg.per_device_eval_batch_size)

    # Configure the optimizer.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay, },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, }, ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

    # Run the training step.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Configure the learning-rate schedule.
    lr_scheduler = get_scheduler(name=cfg.lr_scheduler_type,
                                 optimizer=optimizer,
                                 num_warmup_steps=cfg.num_warmup_steps * cfg.gradient_accumulation_steps,
                                 num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps)

    # Create or use the data loader.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer,
                                                                                            train_dataloader,
                                                                                            eval_dataloader,
                                                                                            lr_scheduler)

    # Run the training step.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # Handle the training checkpoint.
    checkpointing_steps = cfg.checkpointing_steps
    # Validate the required condition.
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Retain this implementation detail from the original training pipeline.
    if cfg.use_logger:
        experiment_config = cfg
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("Language-Based Trajectory Predictor", experiment_config)

    # Compute the evaluation metrics.
    local_rouge_path = "/data/Desktop/BioMiner/Generative_model/rouge/rouge.py"
    metric = evaluate.load(local_rouge_path)
    total_batch_size = cfg.per_device_train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    # Run the training step.
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")

    # Run the training step.
    total_progress_bar = tqdm(
        total=cfg.num_train_epochs,
        desc="总体训练进度",
        position=0,
        leave=True,
        disable=not accelerator.is_local_main_process
    )
    
    # Update the progress display.
    progress_bar = tqdm(range(cfg.max_train_steps), 
                       desc="训练步骤", 
                       position=1, 
                       leave=False,
                       disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0

    # Handle the training checkpoint.
    if cfg.resume_from_checkpoint:
        path = os.path.basename(cfg.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {cfg.resume_from_checkpoint}")
        accelerator.load_state(path)

        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * cfg.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // cfg.gradient_accumulation_stepp

        progress_bar.update(completed_steps)
        total_progress_bar.update(starting_epoch)

    # Configure or use the model.
    best_metric = 0.0
    best_epoch = 0

    # Run the training step.
    for epoch in range(starting_epoch, cfg.num_train_epochs):
        # Update the progress display.
        total_progress_bar.set_description(f"总体训练进度 [Epoch {epoch+1}/{cfg.num_train_epochs}]")
        
        # Update the progress display.
        epoch_progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{cfg.num_train_epochs}",
            position=2,
            leave=False,
            disable=not accelerator.is_local_main_process
        )
        
        model.train()
        if cfg.use_logger:
            total_loss = 0

        # Handle the training checkpoint.
        if cfg.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        # Run the training step.
        for step, batch in enumerate(active_dataloader):
            # Update the progress display.
            epoch_progress_bar.set_description(
                f"Epoch {epoch+1}/{cfg.num_train_epochs} [Step {step+1}/{len(train_dataloader)}]"
            )
            
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                if cfg.use_logger:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update the progress display.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                # Update the progress display.
                epoch_progress_bar.update(1)
                
                # Compute the training loss.
                epoch_progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                    "step": f"{completed_steps}/{cfg.max_train_steps}"
                })

            # Run the training step.
            if completed_steps >= cfg.max_train_steps:
                epoch_progress_bar.close()
                break
        
        # Update the progress display.
        if epoch_progress_bar:
            epoch_progress_bar.close()

        # Run the validation step.
        model.eval()
        
        # Run the validation step.
        val_progress_bar = tqdm(
            total=len(eval_dataloader),
            desc=f"验证进度",
            position=2,
            leave=False,
            disable=not accelerator.is_local_main_process
        )

        # Prepare or report the output.
        example_shown = False

        for step, batch in enumerate(eval_dataloader):
            # Run the validation step.
            val_progress_bar.set_description(f"验证进度 [Batch {step+1}/{len(eval_dataloader)}]")
            
            with torch.no_grad():
                # Collect or process predictions.
                generated_tokens = accelerator.unwrap_model(model).generate(batch["input_ids"],
                                                                            attention_mask=batch["attention_mask"],
                                                                            max_length=cfg.max_target_length,
                                                                            min_length=10,  # Retain this implementation detail from the original training pipeline.
                                                                            length_penalty=2.0,  # Retain this implementation detail from the original training pipeline.
                                                                            num_beams=1)
                generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1,
                                                                    pad_index=tokenizer.pad_token_id)
                labels = batch["labels"]

                # Process the task labels.
                if not cfg.pad_to_max_length:
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                # Run the evaluation step.
                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                # Process the task labels.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                generated_tokens = generated_tokens[0] if isinstance(generated_tokens, tuple) else generated_tokens

                # Process the task labels.
                if not cfg.use_slow_tokenizer:
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                else:
                    filtered_tokens_preds = np.where(generated_tokens >= tokenizer.sp_model.get_piece_size(), 0,
                                                     generated_tokens)
                    decoded_preds = tokenizer.sp_model.decode(filtered_tokens_preds.tolist())
                    filtered_tokens_labels = np.where(labels >= tokenizer.sp_model.get_piece_size(), 0, labels)
                    decoded_labels = tokenizer.sp_model.decode(filtered_tokens_labels.tolist())

                # Prepare or report the output.
                if not example_shown and accelerator.is_main_process:
                    print("\n=== 生成结果示例 ===")
                    print(f"输入 : {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
                    print(f"真实输出 : {decoded_labels[0]}")
                    print(f"模型预测 : {decoded_preds[0]}")
                    print("==================\n")
                    example_shown = True

                # Process the current value.
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            
            # Run the validation step.
            val_progress_bar.update(1)
        
        # Run the validation step.
        val_progress_bar.close()

        # Compute the evaluation metrics.
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        # Run the evaluation step.
        total_progress_bar.set_postfix({
            "rouge1": f"{result.get('rouge1', 0):.2f}%",
            "rouge2": f"{result.get('rouge2', 0):.2f}%", 
            "rougeL": f"{result.get('rougeL', 0):.2f}%",
            "best": f"{best_metric:.2f}%"
        })
        
        logger.info(f"Epoch {epoch} Results: {result}")

        # Retain this implementation detail from the original training pipeline.
        if cfg.use_logger:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

        # Configure or use the model.
        current_metric = result.get("rouge1", 0)
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            # Configure or use the model.
            best_model_path = os.path.join(checkpoint_path, "best_model")
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(best_model_path, is_main_process=accelerator.is_main_process,
                                                save_function=accelerator.save)
                tokenizer.save_pretrained(best_model_path)
                # Configure or use the model.
                best_info = {
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "metrics": result
                }
                with open(os.path.join(best_model_path, "best_model_info.json"), "w") as f:
                    json.dump(best_info, f)
            
            logger.info(f"✅ New best model saved at epoch {epoch} with rouge1: {best_metric:.2f}%")
            
            # Configure or use the model.
            total_progress_bar.write(f"🎉 新最佳模型! Epoch {epoch}, ROUGE-1: {best_metric:.2f}%")

        # Handle the training checkpoint.
        if isinstance(checkpointing_steps, int) and (epoch + 1) % checkpointing_steps == 0:
            checkpoint_path_epoch = os.path.join(checkpoint_path, f"epoch_{epoch}")
            accelerator.save_state(checkpoint_path_epoch)
            logger.info(f"Checkpoint saved at epoch {epoch}")
            
            # Handle the training checkpoint.
            total_progress_bar.write(f"💾 检查点已保存: epoch_{epoch}")
        
        # Update the progress display.
        total_progress_bar.update(1)
        
        # Retain this implementation detail from the original training pipeline.
        total_progress_bar.write(f"✅ Epoch {epoch+1}/{cfg.num_train_epochs} 完成. ROUGE-1: {current_metric:.2f}%")
    
    # Update the progress display.
    total_progress_bar.close()
    
    # Run the training step.
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳模型在 Epoch {best_epoch}, ROUGE-1: {best_metric:.2f}%")
    print("="*60)

    # Configure or use the model.
    accelerator.wait_for_everyone()
    final_model_path = os.path.join(checkpoint_path, "final_model")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(final_model_path, is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_model_path)

        all_results = {f"eval_{k}": v for k, v in result.items()}
        # Configure or use the model.
        all_results["best_metric"] = best_metric
        all_results["best_epoch"] = best_epoch
        with open(os.path.join(final_model_path, "all_results.json"), "w") as f:
            json.dump(all_results, f)

        # Ensure the output directory exists.
        with open(os.path.join(checkpoint_path, "all_results.json"), "w") as f:
            json.dump(all_results, f)

    logger.info(f"Training completed. Best model at epoch {best_epoch} with rouge1: {best_metric}")


if __name__ == "__main__":
    """Documentation for this retained training component."""
    import sys
    import argparse
    import json

    # Retain this implementation detail from the original training pipeline.
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称')

    args = parser.parse_args()

    # Load or validate the configuration.
    config_file = args.cfg if args.cfg else '/data/Desktop/BioMiner/Generative_model/config/config_CNs_LCs_grading.json'
    print(f"尝试加载配置文件: {config_file}")
    print(f"文件是否存在: {os.path.exists(config_file)}")

    # Load or validate the configuration.
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        print("配置文件加载成功")
        print(f"配置项数量: {len(config_dict)}")
        # Load or validate the configuration.
        print("关键配置项:")
        key_items = [
            'dataset_name', 'checkpoint_path', 'checkpoint_name',
            'gradient_accumulation_steps', 'per_device_train_batch_size'
        ]
        for key in key_items:
            if key in config_dict:
                print(f"  {key}: {config_dict[key]}")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        # Load or validate the configuration.
        config_dict = {
            "dataset_name": "eth",
            "checkpoint_path": "./checkpoint/",
            "checkpoint_name": "eth_model",
            "gradient_accumulation_steps": 1,
            "per_device_train_batch_size": 2,
            "max_train_steps": 10,
            "num_train_epochs": 100,
            "learning_rate": 0.0001,
            "seed": 42,
            "use_logger": False,
            "obs_len": 8,
            "pred_len": 12,
            "metric": "pixel",
            "dataset_path": "./datasets/",
            "cache_dir": "./.cache/",
            "tokenizer_name": "./checkpoint/tokenizer/lungcancer-text-bpe/",
            "use_slow_tokenizer": True,
            "history_column": "observation",
            "future_column": "forecast",
            "max_source_length": 256,
            "max_target_length": 128,
            "per_device_eval_batch_size": 2,
            "preprocessing_num_workers": 1,
            "overwrite_cache": True,
            "weight_decay": 0.0,
            "lr_scheduler_type": "linear",
            "num_warmup_steps": 0,
            "model_name_or_path": "./models/t5-small",
            "pad_to_max_length": False,
            "checkpointing_steps": 20,
            "resume_from_checkpoint": None,
            "logger_type": "",
            "num_beams": 1,
            "deterministic": True,
            "top_k": 0,
            "temperature": 1.0,
            "best_of_n": 1,
            "num_samples": 1,
            "per_device_inference_batch_size": 1,
            "model_config_name": None,
            "model_type": None,
            "train": True,
            "eval": False
        }
        print("使用默认配置")


    # Create the required object.
    class DotDict:
        """Documentation for this retained training component."""

        def __init__(self, dictionary):
            self._data = {}
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    self._data[key] = DotDict(value)
                else:
                    self._data[key] = value

        def __getattr__(self, key):
            if key in self._data:
                return self._data[key]
            # Retain this implementation detail from the original training pipeline.
            return None

        def __setattr__(self, key, value):
            if key == '_data':
                # Retain this implementation detail from the original training pipeline.
                super().__setattr__(key, value)
            else:
                if not hasattr(self, '_data'):
                    super().__setattr__('_data', {})
                self._data[key] = value

        def __getitem__(self, key):
            return self._data.get(key, None)

        def __contains__(self, key):
            return key in self._data

        def keys(self):
            return self._data.keys()

        def items(self):
            return self._data.items()

        def values(self):
            return self._data.values()


    # Retain this implementation detail from the original training pipeline.
    config_data = DotDict(config_dict)

    # Load or validate the configuration.
    if args.dataset:
        config_data.dataset_name = args.dataset
        print(f"使用命令行参数设置 dataset_name: {args.dataset}")

    # Load or validate the configuration.
    required_fields = {
        'dataset_name': 'eth',
        'checkpoint_path': './checkpoint/',
        'checkpoint_name': 'eth_model',
        'gradient_accumulation_steps': 1,
        'per_device_train_batch_size': 2,
        'num_train_epochs': 100,
        'checkpointing_steps': 20
    }

    for field, default_value in required_fields.items():
        if not hasattr(config_data, field) or getattr(config_data, field) is None:
            setattr(config_data, field, default_value)
            print(f"设置默认值 {field}: {default_value}")

    # Run the validation step.
    print("\n验证配置:")
    print(f"  dataset_name: {config_data.dataset_name}")
    print(f"  checkpoint_path: {config_data.checkpoint_path}")
    print(f"  checkpoint_name: {config_data.checkpoint_name}")
    print(f"  gradient_accumulation_steps: {config_data.gradient_accumulation_steps}")
    print(f"  per_device_train_batch_size: {config_data.per_device_train_batch_size}")
    print(f"  num_train_epochs: {config_data.num_train_epochs}")
    print(f"  checkpointing_steps: {config_data.checkpointing_steps}")


    cfg = config_data  # Retain this implementation detail from the original training pipeline.
    trainval(cfg)