import os
import re
import json
import numpy as np
import pandas as pd
import sentencepiece as spm
from transformers import T5Tokenizer
from rouge_score import rouge_scorer
import warnings

# Retain this implementation detail from the original pipeline.
warnings.filterwarnings('ignore')

class TokenizerEvaluator:
    def __init__(self, data_path: str, max_samples: int = 4224):
        """Documentation for this retained evaluation component."""
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"未找到数据文件: {data_path}。请确保文件路径正确！")
            
        with open(data_path, 'r', encoding='utf-8') as f: 
            # Retain this implementation detail from the original pipeline.
            self.data = [line.strip() for line in f if line.strip()][:max_samples]
            
        print(f"成功加载真实数据，已严格截取前 {len(self.data)} 条样本用于评估和训练。")
        
        # Load or process the dataset.
        self.temp_train_file = "temp_train_data_4224.txt"
        with open(self.temp_train_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.data))
        
        # Configure or evaluate the tokenizer.
        print("正在加载预训练 T5 Tokenizer...")
        local_t5_path = r"/data/Desktop/BioMiner/Generative_model/models/t5-small"
        self.pretrained_tokenizer = T5Tokenizer.from_pretrained(local_t5_path)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        
    def split_input_output(self, text: str):
        """Documentation for this retained evaluation component."""
        last_q_idx = text.rfind('?')
        if last_q_idx != -1:
            input_text = text[:last_q_idx+1].strip()
            output_text = text[last_q_idx+1:].strip()
            return input_text, output_text
        else:
            return text, ""

    def count_mixed_vocab(self, vocab_list):
        """Documentation for this retained evaluation component."""
        mixed_count = 0
        for piece in vocab_list:
            # Retain this implementation detail from the original pipeline.
            clean_piece = piece.replace(' ', '').replace(' ', '').replace('</w>', '')
            has_alpha = bool(re.search(r'[a-zA-Z]', clean_piece))
            has_digit = bool(re.search(r'\d', clean_piece))
            if has_alpha and has_digit:
                mixed_count += 1
        return mixed_count

    def train_spm(self, model_type: str, vocab_size: int):
        """Documentation for this retained evaluation component."""
        model_prefix = f"temp_spm_{model_type}"
        
        config = {
            'input': self.temp_train_file,  # Run the training-related step.
            'model_prefix': model_prefix,
            'model_type': model_type,
            'character_coverage': 1.0,
            'unk_id': 3, 'bos_id': 1, 'eos_id': 2, 'pad_id': 0,
            'control_symbols': "[PAD],[UNK],[CLS],[SEP],[MASK]",
            'split_by_number': False # Retain this implementation detail from the original pipeline.
        }

        # Load or process the dataset.
        current_vocab_size = vocab_size
        while current_vocab_size > 10:
            config['vocab_size'] = current_vocab_size
            try:
                spm.SentencePieceTrainer.train(**config)
                break
            except Exception as e:
                # Retain this implementation detail from the original pipeline.
                current_vocab_size = int(current_vocab_size * 0.9)
                if current_vocab_size <= 10:
                    raise Exception(f"模型 {model_type} 训练失败，错误信息: {e}")
                
        # Configure or use the model.
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{model_prefix}.model")
        
        # Retain this implementation detail from the original pipeline.
        for ext in ['.model', '.vocab']:
            if os.path.exists(f"{model_prefix}{ext}"):
                os.remove(f"{model_prefix}{ext}")
                
        return sp

    def evaluate_model(self, model_name: str, model_obj, is_pretrained=False):
        """Documentation for this retained evaluation component."""
        # Retain this implementation detail from the original pipeline.
        if is_pretrained:
            vocab = list(model_obj.get_vocab().keys())
            vocab_size = len(vocab)
        else:
            vocab_size = model_obj.vocab_size()
            vocab = [model_obj.IdToPiece(i) for i in range(vocab_size)]
            
        mixed_count = self.count_mixed_vocab(vocab)
        clarity = (1 - mixed_count / vocab_size) * 100 if vocab_size > 0 else 0

        # Retain this implementation detail from the original pipeline.
        covered_sentences = 0
        in_tokens_list, out_tokens_list = [], []
        in_rouge_list, out_rouge_list = [], []

        for sentence in self.data:
            input_text, output_text = self.split_input_output(sentence)
            
            try:
                # Decode the generated sequences.
                if is_pretrained:
                    in_enc = model_obj.encode(input_text)
                    out_enc = model_obj.encode(output_text)
                    in_dec = model_obj.decode(in_enc, skip_special_tokens=True)
                    out_dec = model_obj.decode(out_enc, skip_special_tokens=True)
                    full_dec = model_obj.decode(model_obj.encode(sentence), skip_special_tokens=True)
                else:
                    in_enc = model_obj.EncodeAsIds(input_text)
                    out_enc = model_obj.EncodeAsIds(output_text)
                    in_dec = model_obj.DecodeIds(in_enc)
                    out_dec = model_obj.DecodeIds(out_enc)
                    full_dec = model_obj.DecodeIds(model_obj.EncodeAsIds(sentence))

                # Retain this implementation detail from the original pipeline.
                in_tokens_list.append(len(in_enc))
                out_tokens_list.append(len(out_enc))

                # Compute the required value.
                if in_dec.strip():
                    in_rouge_list.append(self.rouge_scorer.score(input_text, in_dec)['rouge1'].fmeasure)
                else:
                    in_rouge_list.append(0.0)

                if out_dec.strip():
                    out_rouge_list.append(self.rouge_scorer.score(output_text, out_dec)['rouge1'].fmeasure)
                else:
                    out_rouge_list.append(0.0)

                # Retain this implementation detail from the original pipeline.
                if full_dec.replace(' ', '') == sentence.replace(' ', ''):
                    covered_sentences += 1

            except Exception:
                in_tokens_list.append(0)
                out_tokens_list.append(0)
                in_rouge_list.append(0.0)
                out_rouge_list.append(0.0)

        coverage = covered_sentences / len(self.data)

        return {
            'Tokenizer': model_name,
            '#Vocab': vocab_size,
            '#Mixed': mixed_count,
            'Clarity': clarity,
            'Cover': coverage,
            'In_#Token': np.mean(in_tokens_list),
            'In_Rouge': np.mean(in_rouge_list),
            'Out_#Token': np.mean(out_tokens_list),
            'Out_Rouge': np.mean(out_rouge_list)
        }

    def run_all_evaluations(self):
        """Documentation for this retained evaluation component."""
        results = []
        
        # 1. Pretrained
        print("正在评估 Pretrained Tokenizer...")
        res = self.evaluate_model("Pretrained", self.pretrained_tokenizer, is_pretrained=True)
        results.append(res)
        
        # Run the training-related step.
        sp_configs = [
            ("Char", "char", 59),
            ("Word", "word", 13586),
            ("Unigram", "unigram", 1113),
            ("BPE", "bpe", 1224)
        ]
        
        for name, sp_type, target_vocab in sp_configs:
            print(f"正在训练并评估 {name} Tokenizer...")
            sp_model = self.train_spm(sp_type, target_vocab)
            res = self.evaluate_model(name, sp_model, is_pretrained=False)
            results.append(res)
            
        # Run the training-related step.
        if os.path.exists(self.temp_train_file):
            os.remove(self.temp_train_file)
            
        return results


def format_and_save_results(results, output_file="./tokenizer_metric.txt"):
    """Documentation for this retained evaluation component."""
    header = (
        f"{'Tokenizer':<12} | {'#Vocab':>7} | {'#Mixed':>7} | {'Clarity':>7} | {'Cover':>7} | "
        f"{'Input #Token':>12} | {'Input Rouge':>11} | {'Output #Token':>13} | {'Output Rouge':>12}"
    )
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    for r in results:
        line = (
            f"{r['Tokenizer']:<12} | "
            f"{r['#Vocab']:>7d} | "
            f"{r['#Mixed']:>7d} | "
            f"{r['Clarity']:>6.2f}% | "
            f"{r['Cover']:>7.2f} | "
            f"{r['In_#Token']:>12.2f} | "
            f"{r['In_Rouge']:>11.2f} | "
            f"{r['Out_#Token']:>13.2f} | "
            f"{r['Out_Rouge']:>12.2f}"
        )
        lines.append(line)
        
    final_output = "\n".join(lines)
    
    # Report the current status.
    print("\n" + "="*len(header))
    print("评估指标结果汇总 (Tokenizer Metrics)")
    print("="*len(header))
    print(final_output)
    
    # Save the generated artifact.
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output + "\n")
        
    print(f"\n[成功] 结果已保存至: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    # Run the training-related step.
    data_path = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer/tokenizer_train_data_CNs_LCs.txt"
    output_path = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer/tokenizer_metric.txt"
    
    try:
        # Load or process the dataset.
        evaluator = TokenizerEvaluator(data_path, max_samples=4224)
        metrics = evaluator.run_all_evaluations()
        
        # Resolve the required path.
        format_and_save_results(metrics, output_file=output_path)
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()