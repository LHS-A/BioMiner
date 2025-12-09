# Tokenizer training code. Modify the input_file to train the tokenizer with each line as a training sentence.
import os
import sentencepiece as spm
from transformers import T5Tokenizer
import argparse

# Create command-line argument parser
parser = argparse.ArgumentParser()
# Add model type parameter, default is "bpe" (Byte Pair Encoding)
parser.add_argument('--model', default="bpe", type=str, help="Tokenizer model type ('char', 'word', 'unigram', 'bpe').")
# Add tokenizer name parameter, customizable by user
parser.add_argument('--tokenizer_name', default="Generative_model_tokenizer", type=str, help="Tokenizer name (will be used in directory structure).")
args = parser.parse_args()

# Get configuration from command-line arguments
modeltype = args.model  # Model type (e.g., bpe)
tokenizer_name = args.tokenizer_name  # User-defined tokenizer name

# Set tokenizer base directory
tokenizer_basedir = r"/data/Desktop/BioMiner/Generative_model/checkpoint/tokenizer"
# Build filename template
model_dir = f"{tokenizer_name}-text-{modeltype}"
# Create directory: checkpoint/tokenizer/{tokenizer_name}/
base_tokenizer_dir = os.path.join(tokenizer_basedir, tokenizer_name)
os.makedirs(base_tokenizer_dir, exist_ok=True)

# Input file path
input_file = r"/data/Desktop/BioMiner/Generative_model/Train_tokenizer/tokenizer_train_data.txt"

# Train tokenizer model using SentencePiece
spm.SentencePieceTrainer.train(
    # Input file: Text data for tokenizer training
    input=input_file,
    # Model prefix: Name prefix for output model files
    model_prefix=os.path.join(base_tokenizer_dir, model_dir),
    # Vocabulary size: Number of subword units the tokenizer will learn
    vocab_size=1224,
    # Unknown token ID
    unk_id=3,
    # Beginning of sentence token ID
    bos_id=1,
    # End of sentence token ID
    eos_id=2,
    # Padding token ID
    pad_id=0,
    # Control symbols: Special tokens
    control_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]",
    # Model type: Tokenization algorithm used (e.g., bpe)
    model_type=modeltype,
    # Flag for training on extremely large corpus
    train_extremely_large_corpus=True,
    split_by_number=False,  # Preserve number continuity
    # Character coverage: 1.0 means covering all characters
    character_coverage=1.0,
)

# Get trained model file path
vocab_file = os.path.join(base_tokenizer_dir, f"{model_dir}.model")
# Load trained SentencePiece model
sp_model = spm.SentencePieceProcessor()
sp_model.Load(vocab_file)

# Print vocabulary size
print("vocab size:", sp_model.vocab_size())

# Import protobuf model from sentencepiece
from sentencepiece import sentencepiece_model_pb2
# Create model prototype object
m = sentencepiece_model_pb2.ModelProto()
# Read trained model file
with open(vocab_file, 'rb') as f:
    m.ParseFromString(f.read())

# Save model training specifications and vocabulary information to text file
with open(os.path.join(base_tokenizer_dir, f"{model_dir}.txt"), 'w', encoding='utf-8') as f:
    f.write("# trainer_spec\n")
    # Write training specifications
    f.write(m.trainer_spec.__repr__())
    # Clear precompiled character mapping
    m.normalizer_spec.precompiled_charsmap = b''
    f.write("# normalizer_spec\n")
    # Write normalizer specifications
    f.write(m.normalizer_spec.__repr__())
    f.write("# pieces\n")
    # Write all subword units (pieces)
    for piece in m.pieces:
        f.write(piece.piece + '\n')

# Create Hugging Face T5Tokenizer using trained SentencePiece model
tokenizer = T5Tokenizer(vocab_file=vocab_file)
# Save tokenizer to specified directory, making it compatible with Hugging Face Transformers
tokenizer.save_pretrained(base_tokenizer_dir)

print(f"Tokenizer successfully trained and saved to: {base_tokenizer_dir}")
