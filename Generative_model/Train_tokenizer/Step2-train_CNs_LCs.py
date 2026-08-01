# Configure or apply the tokenizer.
import os
import sentencepiece as spm
from transformers import T5Tokenizer
import argparse

# Create the required object.
parser = argparse.ArgumentParser()
# Configure or use the model.
parser.add_argument('--model', default="bpe", type=str, help="Tokenizer model type ('char', 'word', 'unigram', 'bpe').")
# Configure or apply the tokenizer.
parser.add_argument('--tokenizer_name', default="Generative_model_tokenizer_CNs_LCs", type=str, help="Tokenizer name (will be used in directory structure).")
args = parser.parse_args()

# Load or validate the configuration.
modeltype = args.model  # Configure or use the model.
tokenizer_name = args.tokenizer_name  # Configure or apply the tokenizer.

# Configure or apply the tokenizer.
tokenizer_basedir = r"/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer"
# Retain this implementation detail from the original training pipeline.
model_dir = f"{tokenizer_name}-text-{modeltype}"
# Ensure the output directory exists.
base_tokenizer_dir = os.path.join(tokenizer_basedir, tokenizer_name)
os.makedirs(base_tokenizer_dir, exist_ok=True)

# Resolve the required path.
input_file = r"/data/Desktop/BioMiner/Generative_model/Train_tokenizer/tokenizer_train_data_CNs_LCs.txt"

# Configure or apply the tokenizer.
spm.SentencePieceTrainer.train(
    # Configure or apply the tokenizer.
    input=input_file,
    # Configure or use the model.
    model_prefix=os.path.join(base_tokenizer_dir, model_dir),
    # Configure or apply the tokenizer.
    vocab_size=1224,
    # Retain this implementation detail from the original training pipeline.
    unk_id=3,
    # Retain this implementation detail from the original training pipeline.
    bos_id=1,
    # Retain this implementation detail from the original training pipeline.
    eos_id=2,
    # Retain this implementation detail from the original training pipeline.
    pad_id=0,
    # Retain this implementation detail from the original training pipeline.
    control_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]",
    # Configure or use the model.
    model_type=modeltype,
    # Run the training step.
    train_extremely_large_corpus=True,
    split_by_number=False,  # Retain this implementation detail from the original training pipeline.
    # Retain this implementation detail from the original training pipeline.
    character_coverage=1.0,
)

# Run the training step.
vocab_file = os.path.join(base_tokenizer_dir, f"{model_dir}.model")
# Run the training step.
sp_model = spm.SentencePieceProcessor()
sp_model.Load(vocab_file)

# Report the current status.
print("vocab size:", sp_model.vocab_size())

# Configure or use the model.
from sentencepiece import sentencepiece_model_pb2
# Configure or use the model.
m = sentencepiece_model_pb2.ModelProto()
# Run the training step.
with open(vocab_file, 'rb') as f:
    m.ParseFromString(f.read())

# Run the training step.
with open(os.path.join(base_tokenizer_dir, f"{model_dir}.txt"), 'w', encoding='utf-8') as f:
    f.write("# trainer_spec\n")
    # Run the training step.
    f.write(m.trainer_spec.__repr__())
    # Retain this implementation detail from the original training pipeline.
    m.normalizer_spec.precompiled_charsmap = b''
    f.write("# normalizer_spec\n")
    # Write the output data.
    f.write(m.normalizer_spec.__repr__())
    f.write("# pieces\n")
    # Write the output data.
    for piece in m.pieces:
        f.write(piece.piece + '\n')

# Run the training step.
tokenizer = T5Tokenizer(vocab_file=vocab_file)
# Configure or apply the tokenizer.
tokenizer.save_pretrained(base_tokenizer_dir)

print(f"分词器已成功训练并保存到: {base_tokenizer_dir}")
