import os
import json
import pandas as pd
import numpy as np
import sentencepiece as spm
from transformers import T5Tokenizer, AutoTokenizer
from typing import Dict, List, Tuple
import re
from rouge_score import rouge_scorer
import matplotlib
matplotlib.use('Agg')  # Retain this implementation detail from the original training pipeline.
import matplotlib.pyplot as plt
from tqdm import tqdm

class TokenizerEvaluator:
    """Documentation for this retained training component."""
    
    def __init__(self, data_path: str, output_dir: str):
        """Documentation for this retained training component."""
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the required artifact.
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [line.strip() for line in f if line.strip()]
        
        # Run the evaluation step.
        self.sample_size = min(1000, len(self.data))
        self.sample_data = self.data[:self.sample_size]
        
        # Run the training step.
        self.pretrained_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        # Compute the required value.
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        
    def train_tokenizer(self, model_type: str, vocab_size: int = None) -> spm.SentencePieceProcessor:
        """Documentation for this retained training component."""
        # Retain this implementation detail from the original training pipeline.
        if vocab_size is None:
            vocab_sizes = {
                'char': 59,
                'word': 13586,
                'unigram': 1113,
                'bpe': 1224
            }
            vocab_size = vocab_sizes.get(model_type, 1224)
        
        # Resolve the required path.
        model_prefix = os.path.join(self.output_dir, f"temp_{model_type}")
        
        # Load or validate the configuration.
        config = {
            'input': self.data_path,
            'model_prefix': model_prefix,
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': 1.0,
            'unk_id': 3,
            'bos_id': 1,
            'eos_id': 2,
            'pad_id': 0,
            'control_symbols': "[PAD],[UNK],[CLS],[SEP],[MASK]"
        }
        
        # Configure or use the model.
        if model_type == 'char':
            config['model_type'] = 'unigram'  # Retain this implementation detail from the original training pipeline.
            config['split_by_whitespace'] = False
        elif model_type == 'word':
            config['model_type'] = 'word'
            config['split_by_whitespace'] = True
            config['split_by_number'] = False
        else:
            config['split_by_number'] = False
        
        # Configure or apply the tokenizer.
        spm.SentencePieceTrainer.train(**config)
        
        # Run the training step.
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{model_prefix}.model")
        
        # Retain this implementation detail from the original training pipeline.
        for ext in ['.model', '.vocab']:
            temp_file = f"{model_prefix}{ext}"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return sp
    
    def evaluate_vocab_stats(self, sp: spm.SentencePieceProcessor) -> Dict:
        """Documentation for this retained training component."""
        vocab = [sp.IdToPiece(i) for i in range(sp.vocab_size())]
        
        # Retain this implementation detail from the original training pipeline.
        mixed_count = 0
        for piece in vocab:
            if piece.startswith('▁'):
                piece = piece[1:]
            # Validate the required condition.
            has_alpha = bool(re.search(r'[a-zA-Z]', piece))
            has_digit = bool(re.search(r'\d', piece))
            if has_alpha and has_digit:
                mixed_count += 1
        
        # Compute the required value.
        clarity = (1 - mixed_count / len(vocab)) * 100 if vocab else 0
        
        return {
            'vocab_size': len(vocab),
            'mixed_count': mixed_count,
            'clarity': clarity
        }
    
    def evaluate_coverage(self, sp: spm.SentencePieceProcessor) -> float:
        """Documentation for this retained training component."""
        covered_sentences = 0
        
        for sentence in self.sample_data:
            try:
                # Retain this implementation detail from the original training pipeline.
                encoded = sp.EncodeAsPieces(sentence)
                # Retain this implementation detail from the original training pipeline.
                decoded = sp.DecodePieces(encoded)
                if decoded.strip() == sentence.strip():
                    covered_sentences += 1
            except:
                continue
        
        coverage = covered_sentences / len(self.sample_data)
        return coverage
    
    def evaluate_tokenization_efficiency(self, sp: spm.SentencePieceProcessor, 
                                        tokenizer_name: str) -> Dict:
        """Documentation for this retained training component."""
        input_tokens = []
        output_tokens = []
        rouge_scores = []
        
        for sentence in self.sample_data:
            try:
                # Prepare the model input.
                encoded = sp.EncodeAsPieces(sentence)
                input_tokens.append(len(encoded))
                
                # Retain this implementation detail from the original training pipeline.
                reconstructed = sp.DecodePieces(encoded)
                
                # Compute the required value.
                if reconstructed.strip():
                    scores = self.rouge_scorer.score(sentence, reconstructed)
                    rouge_scores.append(scores['rouge1'].fmeasure)
                else:
                    rouge_scores.append(0)
                
                # Prepare or report the output.
                output_tokens.append(len(encoded))  # Prepare the model input.
                
            except Exception as e:
                input_tokens.append(0)
                output_tokens.append(0)
                rouge_scores.append(0)
        
        return {
            'avg_input_tokens': np.mean(input_tokens) if input_tokens else 0,
            'avg_output_tokens': np.mean(output_tokens) if output_tokens else 0,
            'avg_rouge': np.mean(rouge_scores) if rouge_scores else 0
        }
    
    def evaluate_pretrained_tokenizer(self) -> Dict:
        """Documentation for this retained training component."""
        print("评估预训练分词器...")
        
        # Retrieve the required value.
        vocab = self.pretrained_tokenizer.get_vocab()
        
        # Retain this implementation detail from the original training pipeline.
        mixed_count = 0
        for piece in vocab.keys():
            has_alpha = bool(re.search(r'[a-zA-Z]', piece))
            has_digit = bool(re.search(r'\d', piece))
            if has_alpha and has_digit:
                mixed_count += 1
        
        clarity = (1 - mixed_count / len(vocab)) * 100
        
        # Run the evaluation step.
        coverage = self.evaluate_coverage_pretrained()
        
        # Run the evaluation step.
        efficiency = self.evaluate_efficiency_pretrained()
        
        return {
            'vocab_size': len(vocab),
            'mixed_count': mixed_count,
            'clarity': clarity,
            'coverage': coverage,
            'avg_input_tokens': efficiency['avg_input_tokens'],
            'avg_output_tokens': efficiency['avg_output_tokens'],
            'avg_rouge': efficiency['avg_rouge']
        }
    
    def evaluate_coverage_pretrained(self) -> float:
        """Documentation for this retained training component."""
        covered_sentences = 0
        
        for sentence in self.sample_data:
            try:
                encoded = self.pretrained_tokenizer.encode(sentence)
                decoded = self.pretrained_tokenizer.decode(encoded)
                if decoded.strip() == sentence.strip():
                    covered_sentences += 1
            except:
                continue
        
        return covered_sentences / len(self.sample_data)
    
    def evaluate_efficiency_pretrained(self) -> Dict:
        """Documentation for this retained training component."""
        input_tokens = []
        output_tokens = []
        rouge_scores = []
        
        for sentence in self.sample_data:
            try:
                encoded = self.pretrained_tokenizer.encode(sentence)
                input_tokens.append(len(encoded))
                
                reconstructed = self.pretrained_tokenizer.decode(encoded)
                
                if reconstructed.strip():
                    scores = self.rouge_scorer.score(sentence, reconstructed)
                    rouge_scores.append(scores['rouge1'].fmeasure)
                else:
                    rouge_scores.append(0)
                
                output_tokens.append(len(encoded))
                
            except Exception as e:
                input_tokens.append(0)
                output_tokens.append(0)
                rouge_scores.append(0)
        
        return {
            'avg_input_tokens': np.mean(input_tokens) if input_tokens else 0,
            'avg_output_tokens': np.mean(output_tokens) if output_tokens else 0,
            'avg_rouge': np.mean(rouge_scores) if rouge_scores else 0
        }
    
    def run_ablation_study(self):
        """Documentation for this retained training component."""
        print("开始消融实验...")
        
        results = []
        
        # Configure or apply the tokenizer.
        print("\n1. 评估预训练分词器...")
        pretrained_result = self.evaluate_pretrained_tokenizer()
        pretrained_result['tokenizer'] = 'Pretrained'
        results.append(pretrained_result)
        print(f"   完成: Vocab={pretrained_result['vocab_size']}, "
              f"Mixed={pretrained_result['mixed_count']}, "
              f"Clarity={pretrained_result['clarity']:.2f}%")
        
        # Configure or apply the tokenizer.
        print("\n2. 训练和评估字符级别分词器...")
        try:
            char_sp = self.train_tokenizer('char')
            vocab_stats = self.evaluate_vocab_stats(char_sp)
            coverage = self.evaluate_coverage(char_sp)
            efficiency = self.evaluate_tokenization_efficiency(char_sp, 'char')
            
            char_result = {
                'tokenizer': 'Char',
                'vocab_size': vocab_stats['vocab_size'],
                'mixed_count': vocab_stats['mixed_count'],
                'clarity': vocab_stats['clarity'],
                'coverage': coverage,
                'avg_input_tokens': efficiency['avg_input_tokens'],
                'avg_output_tokens': efficiency['avg_output_tokens'],
                'avg_rouge': efficiency['avg_rouge']
            }
            results.append(char_result)
            print(f"   完成: Vocab={char_result['vocab_size']}, "
                  f"Mixed={char_result['mixed_count']}, "
                  f"Clarity={char_result['clarity']:.2f}%")
        except Exception as e:
            print(f"   字符级别分词器评估失败: {e}")
        
        # Configure or apply the tokenizer.
        print("\n3. 训练和评估单词级别分词器...")
        try:
            word_sp = self.train_tokenizer('word')
            vocab_stats = self.evaluate_vocab_stats(word_sp)
            coverage = self.evaluate_coverage(word_sp)
            efficiency = self.evaluate_tokenization_efficiency(word_sp, 'word')
            
            word_result = {
                'tokenizer': 'Word',
                'vocab_size': vocab_stats['vocab_size'],
                'mixed_count': vocab_stats['mixed_count'],
                'clarity': vocab_stats['clarity'],
                'coverage': coverage,
                'avg_input_tokens': efficiency['avg_input_tokens'],
                'avg_output_tokens': efficiency['avg_output_tokens'],
                'avg_rouge': efficiency['avg_rouge']
            }
            results.append(word_result)
            print(f"   完成: Vocab={word_result['vocab_size']}, "
                  f"Mixed={word_result['mixed_count']}, "
                  f"Clarity={word_result['clarity']:.2f}%")
        except Exception as e:
            print(f"   单词级别分词器评估失败: {e}")
        
        # Configure or apply the tokenizer.
        print("\n4. 训练和评估Unigram分词器...")
        try:
            unigram_sp = self.train_tokenizer('unigram')
            vocab_stats = self.evaluate_vocab_stats(unigram_sp)
            coverage = self.evaluate_coverage(unigram_sp)
            efficiency = self.evaluate_tokenization_efficiency(unigram_sp, 'unigram')
            
            unigram_result = {
                'tokenizer': 'Unigram',
                'vocab_size': vocab_stats['vocab_size'],
                'mixed_count': vocab_stats['mixed_count'],
                'clarity': vocab_stats['clarity'],
                'coverage': coverage,
                'avg_input_tokens': efficiency['avg_input_tokens'],
                'avg_output_tokens': efficiency['avg_output_tokens'],
                'avg_rouge': efficiency['avg_rouge']
            }
            results.append(unigram_result)
            print(f"   完成: Vocab={unigram_result['vocab_size']}, "
                  f"Mixed={unigram_result['mixed_count']}, "
                  f"Clarity={unigram_result['clarity']:.2f}%")
        except Exception as e:
            print(f"   Unigram分词器评估失败: {e}")
        
        # Configure or apply the tokenizer.
        print("\n5. 训练和评估BPE分词器...")
        try:
            bpe_sp = self.train_tokenizer('bpe')
            vocab_stats = self.evaluate_vocab_stats(bpe_sp)
            coverage = self.evaluate_coverage(bpe_sp)
            efficiency = self.evaluate_tokenization_efficiency(bpe_sp, 'bpe')
            
            bpe_result = {
                'tokenizer': 'BPE',
                'vocab_size': vocab_stats['vocab_size'],
                'mixed_count': vocab_stats['mixed_count'],
                'clarity': vocab_stats['clarity'],
                'coverage': coverage,
                'avg_input_tokens': efficiency['avg_input_tokens'],
                'avg_output_tokens': efficiency['avg_output_tokens'],
                'avg_rouge': efficiency['avg_rouge']
            }
            results.append(bpe_result)
            print(f"   完成: Vocab={bpe_result['vocab_size']}, "
                  f"Mixed={bpe_result['mixed_count']}, "
                  f"Clarity={bpe_result['clarity']:.2f}%")
        except Exception as e:
            print(f"   BPE分词器评估失败: {e}")
        
        # Retain this implementation detail from the original training pipeline.
        df_results = pd.DataFrame(results)
        
        # Retain this implementation detail from the original training pipeline.
        columns_order = [
            'tokenizer', 'vocab_size', 'mixed_count', 'clarity', 'coverage',
            'avg_input_tokens', 'avg_rouge', 'avg_output_tokens'
        ]
        df_results = df_results[columns_order]
        
        # Prepare or report the output.
        df_display = df_results.copy()
        df_display['clarity'] = df_display['clarity'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        df_display['coverage'] = df_display['coverage'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        df_display['avg_input_tokens'] = df_display['avg_input_tokens'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        df_display['avg_output_tokens'] = df_display['avg_output_tokens'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        df_display['avg_rouge'] = df_display['avg_rouge'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
        # Save the generated artifact.
        output_csv = os.path.join(self.output_dir, "tokenizer_evaluation_results.csv")
        df_results.to_csv(output_csv, index=False)
        
        # Retain this implementation detail from the original training pipeline.
        latex_table = self.generate_latex_table(df_results)
        
        # Save the generated artifact.
        latex_file = os.path.join(self.output_dir, "tokenizer_evaluation_table.tex")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        # Create the required object.
        self.create_visualizations(df_results)
        
        print(f"\n{'='*80}")
        print("消融实验完成!")
        print(f"{'='*80}")
        print("\n评估结果:")
        print(df_display.to_string(index=False))
        print(f"\n详细结果已保存到: {output_csv}")
        print(f"LaTeX表格已保存到: {latex_file}")
        
        return df_results
    
    def generate_latex_table(self, df_results: pd.DataFrame) -> str:
        """Documentation for this retained training component."""
        
        # Retain this implementation detail from the original training pipeline.
        def format_value(val, is_percent=False, is_float=False):
            if pd.isnull(val):
                return "N/A"
            if is_percent:
                return f"{val:.2f}"
            if is_float:
                return f"{val:.4f}"
            if isinstance(val, float):
                return f"{val:.2f}"
            return str(val)
        
        # Retain this implementation detail from the original training pipeline.
        rows = []
        for _, row in df_results.iterrows():
            tokenizer = f"\\textbf{{{row['tokenizer']}}}" if row['tokenizer'] == 'BPE' else f"\\textbf{{{row['tokenizer']}}}" if row['tokenizer'] == 'Pretrained' else row['tokenizer']
            
            row_str = f"{tokenizer} & "
            row_str += f"{int(row['vocab_size'])} & "
            row_str += f"{int(row['mixed_count'])} & "
            row_str += f"{format_value(row['clarity'], is_percent=True)} & "
            row_str += f"{format_value(row['coverage'], is_percent=True)} & "
            row_str += f"{format_value(row['avg_input_tokens'])} & "
            row_str += f"{format_value(row['avg_rouge'], is_float=True)} & "
            row_str += f"{format_value(row['avg_output_tokens'])} & "
            row_str += f"{format_value(row['avg_rouge'], is_float=True)} \\\\"
            rows.append(row_str)
        
        # Retain this implementation detail from the original training pipeline.
        latex = r"""\begin{table}[!t]
\centering
\caption{Evaluation of the tokenizer characteristics. \#Vocab: the total number of unique words in the tokenizer, \#Mixed: The number of unique entries that contain both characters and numerals, Clarity: Percentage of non-mixed cases in vocab, Cover: Coverage of the tokenizer that can cover all sentences in the dataset, \#Token: The average number of tokens per sentence. Rouge: ROUGE-1 score between the original sentences and their reconstructed ones after tokenization.}
\label{tab:tokenizer_eval}
\resizebox{0.95\columnwidth}{!}{%
\begin{tabular}{c| c c c c| c c| c c}
\toprule
\multirow{2}{*}{\textbf{Tokenizer}} & \multicolumn{4}{c}{\textbf{Summary}} & \multicolumn{2}{c}{\textbf{Input sentence}} & \multicolumn{2}{c}{\textbf{Output sentence}} \\
\cline{2-9}
 & \textbf{\#Vocab} & \textbf{\#Mixed} & \textbf{Clarity} & \textbf{Cover} & \textbf{\#Token} & \textbf{Rouge} & \textbf{\#Token} & \textbf{Rouge} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}"""
        
        return latex
    
    def create_visualizations(self, df_results: pd.DataFrame):
        """Documentation for this retained training component."""
        
        # Retain this implementation detail from the original training pipeline.
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Retain this implementation detail from the original training pipeline.
        ax1 = axes[0, 0]
        tokenizers = df_results['tokenizer']
        vocab_sizes = df_results['vocab_size']
        mixed_counts = df_results['mixed_count']
        
        x = np.arange(len(tokenizers))
        width = 0.35
        
        ax1.bar(x - width/2, vocab_sizes, width, label='Vocab Size', color='skyblue')
        ax1.bar(x + width/2, mixed_counts, width, label='Mixed Count', color='lightcoral')
        
        ax1.set_xlabel('Tokenizer')
        ax1.set_ylabel('Count')
        ax1.set_title('Vocabulary Size vs Mixed Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tokenizers, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Retain this implementation detail from the original training pipeline.
        ax2 = axes[0, 1]
        clarity = df_results['clarity']
        coverage = df_results['coverage'] * 100  # Retain this implementation detail from the original training pipeline.
        
        ax2.bar(x - width/2, clarity, width, label='Clarity (%)', color='lightgreen')
        ax2.bar(x + width/2, coverage, width, label='Coverage (%)', color='gold')
        
        ax2.set_xlabel('Tokenizer')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Clarity and Coverage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tokenizers, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Retain this implementation detail from the original training pipeline.
        ax3 = axes[1, 0]
        input_tokens = df_results['avg_input_tokens']
        output_tokens = df_results['avg_output_tokens']
        
        ax3.bar(x - width/2, input_tokens, width, label='Input Tokens', color='violet')
        ax3.bar(x + width/2, output_tokens, width, label='Output Tokens', color='orange')
        
        ax3.set_xlabel('Tokenizer')
        ax3.set_ylabel('Average Tokens')
        ax3.set_title('Tokenization Efficiency')
        ax3.set_xticks(x)
        ax3.set_xticklabels(tokenizers, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Retain this implementation detail from the original training pipeline.
        ax4 = axes[1, 1]
        rouge_scores = df_results['avg_rouge']
        
        bars = ax4.bar(x, rouge_scores, color='steelblue')
        ax4.set_xlabel('Tokenizer')
        ax4.set_ylabel('ROUGE-1 Score')
        ax4.set_title('Reconstruction Quality (ROUGE)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(tokenizers, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Process the task labels.
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the generated artifact.
        chart_path = os.path.join(self.output_dir, "tokenizer_evaluation_charts.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {chart_path}")

def main():
    """Documentation for this retained training component."""
    
    # Resolve the required path.
    data_path = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer/tokenizer_train_data.txt"
    output_dir = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer"
    
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the evaluation step.
    print("初始化分词器评估器...")
    evaluator = TokenizerEvaluator(data_path, output_dir)
    
    print(f"数据路径: {data_path}")
    print(f"数据大小: {len(evaluator.data)} 个句子")
    print(f"采样大小: {evaluator.sample_size} 个句子")
    
    # Retain this implementation detail from the original training pipeline.
    results = evaluator.run_ablation_study()
    
    # Report the current status.
    print("\n" + "="*80)
    print("汇总统计:")
    print("="*80)
    
    best_clarity = results.loc[results['clarity'].idxmax()]
    best_coverage = results.loc[results['coverage'].idxmax()]
    best_efficiency = results.loc[results['avg_input_tokens'].idxmin()]
    best_rouge = results.loc[results['avg_rouge'].idxmax()]
    
    print(f"最佳清晰度: {best_clarity['tokenizer']} ({best_clarity['clarity']:.2f}%)")
    print(f"最佳覆盖率: {best_coverage['tokenizer']} ({best_coverage['coverage']:.4f})")
    print(f"最佳效率 (最少token数): {best_efficiency['tokenizer']} ({best_efficiency['avg_input_tokens']:.2f} tokens/sentence)")
    print(f"最佳重建质量: {best_rouge['tokenizer']} (ROUGE-1: {best_rouge['avg_rouge']:.4f})")

if __name__ == "__main__":
    # Retain this implementation detail from the original training pipeline.
    try:
        import rouge_score
    except ImportError:
        print("正在安装rouge-score...")
        import subprocess
        subprocess.check_call(["pip", "install", "rouge-score"])
    
    try:
        import sentencepiece
    except ImportError:
        print("正在安装sentencepiece...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentencepiece"])
    
    try:
        import transformers
    except ImportError:
        print("正在安装transformers...")
        import subprocess
        subprocess.check_call(["pip", "install", "transformers"])
    
    try:
        main()
    except Exception as e:
        print(f"运行消融实验时出错: {e}")
        import traceback
        traceback.print_exc()