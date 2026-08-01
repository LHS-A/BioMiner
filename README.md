# BioMiner

Paper-aligned code for simultaneous four-level grading of corneal nerve tortuosity and Langerhans cell activation from corneal confocal microscopy.

The repository follows the three sequential components in the Methods section exactly:

1. **Vision** — topology-aware reconstruction pre-training, then topology-informed grading adaptation.
2. **Text** — numerical-aware tokenization, generative clinical-QA pre-training, then semantics-informed grading adaptation.
3. **Fusion** — query-guided cross-modal alignment, followed by bi-directional semantic calibration and joint grading.

The JSON values in `examples/` are synthetic interface examples, not study observations or clinically meaningful thresholds.

## Method-aligned file tree

```text
BioMiner/
├── run_biominer.py                         # Unified entry point
├── vision_branch/
│   ├── backbone.py                         # Shared ResNet-50 encoder
│   ├── topology_aware_pretraining/
│   │   ├── topology.py                     # NSP extraction, BFS tracing, corruption masks
│   │   ├── dataset.py                      # Image + nerve/cell segmentation inputs
│   │   ├── model.py                        # Reconstruction model, L_TPLC and L_TGIC
│   │   └── train.py
│   └── grading_adaptation/
│       ├── dataset.py
│       ├── model.py                        # GAP + two four-level classifiers
│       └── train.py
├── text_branch/
│   ├── clinical_narrative.py               # Table 2 narrative and 12-feature schema
│   ├── validate_input.py
│   ├── numerical_tokenizer/
│   │   ├── prepare_corpus.py
│   │   ├── train.py                        # Numerical-aware SentencePiece BPE
│   │   └── evaluate.py
│   ├── generative_pretraining/
│   │   ├── config.json
│   │   ├── train.py                        # T5 clinical-QA pre-training
│   │   └── evaluate.py                     # ROUGE-1/2/L
│   └── grading_adaptation/
│       ├── dataset.py
│       ├── encoder.py                      # Adapted T5 encoder exposed to Fusion
│       ├── model.py                        # Shared MLP + two four-level classifiers
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
├── fusion_branch/
│   ├── dataset.py                          # Paired image, masks, and clinical narrative
│   ├── model.py                            # Shared queries + symmetric cross-modal attention
│   └── train.py                            # Frozen Vision/Text encoders
├── evaluation/
│   ├── metrics.py                          # Level-wise ACC and one-vs-rest AUC
│   └── evaluate_predictions.py
├── examples/
└── tests/
```

## Installation

Python 3.10 or newer and an NVIDIA GPU are recommended. The reported experiments used one RTX 3090.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Unified command interface

All public stages are available through one entry point:

```bash
python run_biominer.py --help
python run_biominer.py COMMAND --help
```

Available commands are `vision-pretrain`, `vision-adapt`, `tokenizer-prepare`, `tokenizer-train`, `tokenizer-evaluate`, `text-pretrain`, `text-pretrain-evaluate`, `text-adapt`, `text-evaluate`, `text-predict`, `fusion-train`, `evaluate-predictions`, and `validate-text`.

To execute several stages sequentially, edit [`examples/pipeline.example.json`](examples/pipeline.example.json) and run:

```bash
python run_biominer.py pipeline --config examples/pipeline.example.json
```

## Data contracts

Use participant-level five-fold splits: three folds for training, one for validation, and one for internal testing. Keep external cohorts fully withheld. Never place images from the same participant in different splits.

- Vision manifests follow [`examples/vision_manifest.example.json`](examples/vision_manifest.example.json).
- Text samples follow [`examples/text_samples.json`](examples/text_samples.json).
- Paired multimodal records follow [`examples/fusion_manifest.example.json`](examples/fusion_manifest.example.json).
- Prediction probabilities follow [`examples/predictions.example.json`](examples/predictions.example.json).

Validate and normalize structured text without images or model weights:

```bash
python run_biominer.py validate-text examples/text_samples.json \
  --output examples/text_samples.normalized.json
```

## 1. Vision branch

### 1.1 Topology-aware reconstruction pre-training

```bash
python run_biominer.py vision-pretrain \
  --manifest data/fold1/vision_pretrain.json \
  --root data \
  --output outputs/fold1/vision/topology_autoencoder.pt
```

The defaults reproduce the reported topology settings: 12 non-simple-point seeds, six per anatomical structure; uniformly sampled segment lengths of 60–100 pixels; 7×7 image-mask dilation; 11×11 label-mask dilation; zero-mean Gaussian replacement; ResNet-50; and AdamW at `1e-3`. The Gaussian standard deviation is configurable because the paper does not report it.

### 1.2 Topology-informed grading adaptation

```bash
python run_biominer.py vision-adapt \
  --manifest data/fold1/vision_train.json \
  --root data \
  --pretrained outputs/fold1/vision/topology_autoencoder.pt \
  --output outputs/fold1/vision/grader.pt
```

This stage discards the reconstruction decoder, retains the encoder, applies global average pooling, and jointly optimizes two independent four-level heads with cross-entropy.

## 2. Text branch

### 2.1 Numerical-aware tokenizer

Prepare one-line clinical-QA sentences and train SentencePiece without splitting numbers:

```bash
python run_biominer.py tokenizer-prepare data/domain_knowledge.json data/numerical_qa.json \
  --output outputs/text_branch/tokenizer_corpus.txt

python run_biominer.py tokenizer-train \
  --corpus outputs/text_branch/tokenizer_corpus.txt \
  --output-dir outputs/text_branch/numerical_tokenizer

python run_biominer.py tokenizer-evaluate \
  --tokenizer outputs/text_branch/numerical_tokenizer \
  --corpus outputs/text_branch/tokenizer_corpus.txt
```

### 2.2 Generative clinical-QA pre-training

The input directory must contain `<dataset_name>_train.json` and `<dataset_name>_val.json`, each with `observation` and `forecast` fields. Edit [`text_branch/generative_pretraining/config.json`](text_branch/generative_pretraining/config.json), then run:

```bash
python run_biominer.py text-pretrain \
  --config text_branch/generative_pretraining/config.json
```

The paper configuration is T5 with the numerical tokenizer, AdamW, 200 epochs, batch size 64, and learning rate `1e-4`.

### 2.3 Semantics-informed grading adaptation

```bash
python run_biominer.py text-adapt \
  --train-json data/fold1/text_train.json \
  --val-json data/fold1/text_val.json \
  --base-model models/t5-clinical-base \
  --tokenizer outputs/text_branch/numerical_tokenizer \
  --generative-checkpoint outputs/text_branch/generative_pretraining/best_model \
  --output outputs/fold1/text/grader.pt
```

The encoder receives the unified clinical narrative plus the D5 functional-grading question. Masked mean pooling and a shared MLP produce task-adapted features for the two classification heads and subsequent fusion.

## 3. Fusion branch

```bash
python run_biominer.py fusion-train \
  --manifest data/fold1/fusion_train.json \
  --image-root data \
  --vision-checkpoint outputs/fold1/vision/grader.pt \
  --text-model models/t5-clinical-base \
  --text-checkpoint outputs/fold1/text/grader.pt \
  --tokenizer outputs/text_branch/numerical_tokenizer \
  --output outputs/fold1/fusion/model.pt
```

The adapted Vision and Text encoders are frozen. Shared learnable queries independently attend to local/global visual tokens and textual tokens. Symmetric residual cross-attention then calibrates both modalities. Concatenated calibrated tokens are pooled and passed to both task-specific classifiers. Fusion uses AdamW, batch size 64, and learning rate `1e-4`.

## Evaluation

```bash
python run_biominer.py evaluate-predictions examples/predictions.example.json
```

The evaluator reports overall ACC, four within-level correct-classification rates, four one-vs-rest AUCs, and macro-AUC. Report five-fold results as mean ± standard deviation. The paper compares methods with two-sided Wilcoxon signed-rank tests and Benjamini–Hochberg correction.

## Verification

```bash
python -m compileall -q vision_branch text_branch fusion_branch evaluation tests run_biominer.py
python -m unittest discover -s tests -t . -v
python run_biominer.py validate-text examples/text_samples.json
```

## Reproduction boundary

Exact numerical reproduction additionally requires the original participant-level fold manifests, CCM images, nerve/cell segmentation masks, trained base model, numerical tokenizer corpus, clinical-QA corpus, and study checkpoints. These restricted artifacts are intentionally not fabricated in this repository.
