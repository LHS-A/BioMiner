# This script is the core training module for text generation, fine-tuning a sequence-to-sequence model on a
# Langerhans-cell–centric dataset formatted as question–answer pairs. Each sample contains context, question,
# and answer triplets.
# The goal is to produce a domain-expert model specialized in Langerhans-cell biology.
# Before running, edit the parameters in config/config.json under the project root, then execute this file.
# TODO: To launch training, simply change the json path that follows the `else` statement below
# (line 581: config_file = args.cfg if args.cfg else '.../config_LCs_grading.json').
# All hyper-parameters are managed in that single file; no other code changes are required.
# Optional: adjust the ROUGE script path on line 273 if necessary: local_rouge_path = ".../rouge.py"

import sys
import os

# Get the parent directory of the current file (project root)
current_dir = os.path.dirname(os.path.abspath(____))
parent_dir = os.path.dirname(current_dir)
# Add the project root to PYTHONPATH so custom modules can be imported
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import logging
import math
import random

import datasets
import evaluate
import numpy as np
import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

from Generative_model.utils.nltoolkit import init_nltk, postprocess_text

logger = get_logger(__name__)


def trainval(cfg):
    """
    Main training and validation entry point.
    Executes the full training loop and evaluation protocol.

    Args:
        cfg: Configuration object holding all training hyper-parameters.
    """

    print("=== Configuration Debug ===")
    print(f"checkpoint_path type: {type(cfg.checkpoint_path)}, value: {cfg.checkpoint_path}")
    print(f"checkpoint_name type: {type(cfg.checkpoint_name)}, value: {cfg.checkpoint_name}")
    print(f"dataset_name type: {type(cfg.dataset_name)}, value: {cfg.dataset_name}")
    print("=========================")

    if cfg.checkpoint_name is None:
        cfg.checkpoint_name = "default_model"
        print("Warning: checkpoint_name was None, set to 'default_model'")

    if cfg.checkpoint_path is None:
        cfg.checkpoint_path = "./checkpoint/"
        print("Warning: checkpoint_path was None, set to './checkpoint/'")

    init_nltk()

    checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    accelerator_log_kwargs = {}
    if cfg.use_logger:
        accelerator_log_kwargs["log_with"] = cfg.logger_type
        accelerator_log_kwargs["project_dir"] = checkpoint_path

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
    accelerator.wait_for_everyone()

    preprocessed_train_dataset_name = f"{cfg.dataset_name}_train.json"
    preprocessed_val_dataset_name = f"{cfg.dataset_name}_val.json"
    preprocessed_dataset_path = os.path.join(cfg.dataset_path)

    data_files = {}
    data_files["train"] = os.path.join(preprocessed_dataset_path, preprocessed_train_dataset_name)
    data_files["validation"] = os.path.join(preprocessed_dataset_path, preprocessed_val_dataset_name)

    if not os.path.exists(data_files["train"]) or not os.path.exists(data_files["validation"]):
        raise ValueError(
            f"Pre-processed dataset files not found: {data_files['train']} or {data_files['validation']}. "
            "Please run `./script/preprocessor.sh` first."
        )

    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    if cfg.model_config_name or cfg.model_name_or_path:
        config = AutoConfig.from_pretrained(
            cfg.model_config_name if cfg.model_config_name else cfg.model_name_or_path,
            trust_remote_code=False,
            cache_dir=cfg.cache_dir,
        )
    else:
        config = CONFIG_MAPPING[cfg.model_type]()
        logger.warning("Instantiating a brand-new config from scratch.")

    if cfg.tokenizer_name or cfg.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name_or_path,
            trust_remote_code=False,
            cache_dir=cfg.cache_dir,
            use_fast=not cfg.use_slow_tokenizer,
        )
    else:
        raise ValueError(
            "Instantiating a new tokenizer from scratch is not supported by this script. "
            "Use utils/tokenizer to create and save one, then load via --tokenizer_name."
        )

    if cfg.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_name_or_path,
            config=config,
            trust_remote_code=False,
            cache_dir=cfg.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=False)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Ensure `config.decoder_start_token_id` is correctly defined")
    if cfg.tokenizer_name is not None:
        model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names

    history_column = cfg.history_column
    if history_column not in column_names:
        raise ValueError(
            f"--history_column value '{cfg.history_column}' must be one of: {', '.join(column_names)}"
        )
    future_column = cfg.future_column
    if future_column not in column_names:
        raise ValueError(
            f"--future_column value '{cfg.future_column}' must be one of: {', '.join(column_names)}"
        )

    padding = "max_length" if cfg.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[history_column]
        targets = examples[future_column]

        model_inputs = tokenizer(
            inputs,
            max_length=cfg.max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=cfg.max_target_length,
            padding=padding,
            truncation=True,
        )

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=cfg.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="Tokenizing train dataset",
        )

        val_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=cfg.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not cfg.overwrite_cache,
            desc="Tokenizing validation dataset",
        )

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Training sample {index}: {train_dataset[index]}.")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=cfg.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=cfg.per_device_eval_batch_size,
    )

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = cfg.checkpointing_steps
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if cfg.use_logger:
        experiment_config = cfg
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("Language-Based Trajectory Predictor", experiment_config)

    local_rouge_path = "/data/Desktop/BioMiner/Generative_model/rouge/rouge.py"
    metric = evaluate.load(local_rouge_path)
    total_batch_size = (
        cfg.per_device_train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")

    total_progress_bar = tqdm(
        total=cfg.num_train_epochs,
        desc="Overall training progress",
        position=0,
        leave=True,
        disable=not accelerator.is_local_main_process,
    )

    progress_bar = tqdm(
        range(cfg.max_train_steps),
        desc="Training steps",
        position=1,
        leave=False,
        disable=not accelerator.is_local_main_process,
    )

    completed_steps = 0
    starting_epoch = 0

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
            completed_steps = resume_step // cfg.gradient_accumulation_steps

        progress_bar.update(completed_steps)
        total_progress_bar.update(starting_epoch)

    best_metric = 0.0
    best_epoch = 0

    for epoch in range(starting_epoch, cfg.num_train_epochs):
        total_progress_bar.set_description(f"Overall training progress [Epoch {epoch+1}/{cfg.num_train_epochs}]")

        epoch_progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{cfg.num_train_epochs}",
            position=2,
            leave=False,
            disable=not accelerator.is_local_main_process,
        )

        model.train()
        if cfg.use_logger:
            total_loss = 0

        if cfg.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
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

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                epoch_progress_bar.update(1)
                epoch_progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                        "step": f"{completed_steps}/{cfg.max_train_steps}",
                    }
                )

            if completed_steps >= cfg.max_train_steps:
                epoch_progress_bar.close()
                break

        if epoch_progress_bar:
            epoch_progress_bar.close()

        model.eval()

        val_progress_bar = tqdm(
            total=len(eval_dataloader),
            desc="Validation progress",
            position=2,
            leave=False,
            disable=not accelerator.is_local_main_process,
        )

        example_shown = False

        for step, batch in enumerate(eval_dataloader):
            val_progress_bar.set_description(f"Validation progress [Batch {step+1}/{len(eval_dataloader)}]")

            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=cfg.max_target_length,
                    min_length=10,
                    length_penalty=2.0,
                    num_beams=1,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                if not cfg.pad_to_max_length:
                    labels = accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                    )

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                generated_tokens = generated_tokens[0] if isinstance(generated_tokens, tuple) else generated_tokens

                if not cfg.use_slow_tokenizer:
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                else:
                    filtered_tokens_preds = np.where(
                        generated_tokens >= tokenizer.sp_model.get_piece_size(), 0, generated_tokens
                    )
                    decoded_preds = tokenizer.sp_model.decode(filtered_tokens_preds.tolist())
                    filtered_tokens_labels = np.where(labels >= tokenizer.sp_model.get_piece_size(), 0, labels)
                    decoded_labels = tokenizer.sp_model.decode(filtered_tokens_labels.tolist())

                if not example_shown and accelerator.is_main_process:
                    print("\n=== Generation Example ===")
                    print(f"Input : {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
                    print(f"Target: {decoded_labels[0]}")
                    print(f"Predicted: {decoded_preds[0]}")
                    print("==========================\n")
                    example_shown = True

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            val_progress_bar.update(1)

        val_progress_bar.close()

        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        total_progress_bar.set_postfix(
            {
                "rouge1": f"{result.get('rouge1', 0):.2f}%",
                "rouge2": f"{result.get('rouge2', 0):.2f}%",
                "rougeL": f"{result.get('rougeL', 0):.2f}%",
                "best": f"{best_metric:.2f}%",
            }
        )
        logger.info(f"Epoch {epoch} Results: {result}")

        if cfg.use_logger:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

        current_metric = result.get("rouge1", 0)
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch

            best_model_path = os.path.join(checkpoint_path, "best_model")
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    best_model_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                tokenizer.save_pretrained(best_model_path)

                best_info = {
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "metrics": result,
                }
                with open(os.path.join(best_model_path, "best_model_info.json"), "w") as f:
                    json.dump(best_info, f)

            logger.info(f"✅ New best model saved at epoch {epoch} with rouge1: {best_metric:.2f}%")
            total_progress_bar.write(f"🎉 New best model! Epoch {epoch}, ROUGE-1: {best_metric:.2f}%")

        if isinstance(checkpointing_steps, int) and (epoch + 1) % checkpointing_steps == 0:
            checkpoint_path_epoch = os.path.join(checkpoint_path, f"epoch_{epoch}")
            accelerator.save_state(checkpoint_path_epoch)
            logger.info(f"Checkpoint saved at epoch {epoch}")
            total_progress_bar.write(f"💾 Checkpoint saved: epoch_{epoch}")

        total_progress_bar.update(1)
        total_progress_bar.write(f"✅ Epoch {epoch+1}/{cfg.num_train_epochs} finished. ROUGE-1: {current_metric:.2f}%")

    total_progress_bar.close()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model at Epoch {best_epoch}, ROUGE-1: {best_metric:.2f}%")
    print("=" * 60)

    accelerator.wait_for_everyone()
    final_model_path = os.path.join(checkpoint_path, "final_model")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        final_model_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_model_path)

        all_results = {f"eval_{k}": v for k, v in result.items()}
        all_results["best_metric"] = best_metric
        all_results["best_epoch"] = best_epoch
        with open(os.path.join(final_model_path, "all_results.json"), "w") as f:
            json.dump(all_results, f)

        with open(os.path.join(checkpoint_path, "all_results.json"), "w") as f:
            json.dump(all_results, f)

    logger.info(f"Training completed. Best model at epoch {best_epoch} with rouge1: {best_metric}")


if __name__ == "__main__":
    """
    Main entry point: parses CLI arguments and starts training.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to configuration file")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    args = parser.parse_args()

    config_file = args.cfg if args.cfg else "/data/Desktop/BioMiner/Generative_model/config/config_LCs_grading.json"
    print(f"Attempting to load config: {config_file}")
    print(f"File exists: {os.path.exists(config_file)}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        print("Config loaded successfully")
        print(f"Number of entries: {len(config_dict)}")
        key_items = [
            "dataset_name",
            "checkpoint_path",
            "checkpoint_name",
            "gradient_accumulation_steps",
            "per_device_train_batch_size",
        ]
        print("Key entries:")
        for key in key_items:
            if key in config_dict:
                print(f"  {key}: {config_dict[key]}")
    except Exception as e:
        print(f"Failed to load config: {e}")
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
            "eval": False,
        }
        print("Using default configuration")

    class DotDict:
        """
        Converts a nested dictionary into an object whose attributes can be accessed via dot notation.
        """

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
            return None

        def __setattr__(self, key, value):
            if key == "_data":
                super().__setattr__(key, value)
            else:
                if not hasattr(self, "_data"):
                    super().__setattr__("_data", {})
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

    config_data = DotDict(config_dict)

    if args.dataset:
        config_data.dataset_name = args.dataset
        print(f"Dataset name overridden via CLI: {args.dataset}")

    required_fields = {
        "dataset_name": "eth",
        "checkpoint_path": "./checkpoint/",
        "checkpoint_name": "eth_model",
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": 2,
        "num_train_epochs": 100,
        "checkpointing_steps": 20,
    }

    for field, default_value in required_fields.items():
        if not hasattr(config_data, field) or getattr(config_data, field) is None:
            setattr(config_data, field, default_value)
            print(f"Set default {field}: {default_value}")

    print("\nConfiguration sanity check:")
    print(f"  dataset_name: {config_data.dataset_name}")
    print(f"  checkpoint_path: {config_data.checkpoint_path}")
    print(f"  checkpoint_name: {config_data.checkpoint_name}")
    print(f"  gradient_accumulation_steps: {config_data.gradient_accumulation_steps}")
    print(f"  per_device_train_batch_size: {config_data.per_device_train_batch_size}")
    print(f"  num_train_epochs: {config_data.num_train_epochs}")
    print(f"  checkpointing_steps: {config_data.checkpointing_steps}")

    cfg = config_data
    trainval(cfg)
