# BioMiner reproduction

This repository implements the method described in **BioMiner: A Unified Language-Driven Framework for Automated and Standardized Assessment of Langerhans Cell Activation**. It performs simultaneous four-level grading of corneal nerve tortuosity and Langerhans cell activation through three sequential components:

1. topology-aware visual representation learning;
2. numerical semantic reasoning with a numerical-aware tokenizer and T5;
3. bi-directional feature-level cross-modal alignment.

The example measurements and labels in `examples/` are synthetic interface examples. They are not observations from the paper, are not suitable for clinical use, and must not be used to reproduce the paper's reported results.

## Project layout

```text
biominer/
  topology.py       Digital-topology corruption and paper losses
  vision.py         ResNet-50 autoencoder and dual grading model
  data.py           Twelve-morphometric text contract and fusion dataset
  text.py           Adapter for the task-fine-tuned T5 encoder
  fusion.py         Shared-query alignment and bi-directional calibration
  metrics.py        Within-level ACC and one-vs-rest AUC
Generative_model/   Original dual-task generative and tokenizer training logic
scripts/            Training, validation, and evaluation entry points
examples/           Synthetic JSON contracts
tests/              Lightweight method and interface tests
```

## Installation

Python 3.10 or newer and an NVIDIA GPU are recommended. The paper used one RTX 3090.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Validate the text-only example

This command requires no images, model weights, or downloads:

```bash
python -m scripts.validate_text_input examples/text_samples.json --output examples/text_samples.normalized.json
```

Each structured sample contains patient context and all 12 continuous measurements from Table 2. The validator renders the unified clinical narrative followed by the D5 functional-grading question. A sample may instead supply a ready-made `input` string plus `nerve_label` and `cell_label`.

## Required real-data manifests

Use patient-level splits. For each of the paper's five internal folds, use three folds for training, one for validation, and one for testing. Keep all external cohorts completely withheld. Never place images from one participant in more than one split.

The image manifest follows [examples/image_manifest.example.json](examples/image_manifest.example.json). Paths are relative to `--root` or `--image-root`. Every record must point to the original grayscale CCM image, its binary nerve mask, its binary cell mask, and the two grades in `{0,1,2,3}`. Fusion JSON additionally includes either the structured text fields shown in [examples/text_samples.json](examples/text_samples.json) or a pre-rendered `input` narrative.

## 1. Topology-aware visual pre-training

```bash
python -m scripts.train_topology_pretraining \
  --manifest data/fold1/pretrain.json \
  --root data \
  --output checkpoints/fold1/topology_autoencoder.pt
```

The defaults reproduce the reported critical settings: 12 NSP seeds split equally between nerve and cell skeletons, segment lengths uniformly sampled from 60 to 100 pixels, 7x7 image-mask dilation, 11x11 label-mask dilation, ResNet-50, AdamW, and learning rate `1e-3`. Corrupted image pixels receive zero-mean Gaussian noise. The noise standard deviation is exposed because the paper does not report its value.

## 2. Vision grading adaptation

```bash
python -m scripts.finetune_vision \
  --manifest data/fold1/vision_train.json \
  --root data \
  --pretrained checkpoints/fold1/topology_autoencoder.pt \
  --output checkpoints/fold1/vision_grader.pt
```

The reconstruction decoder is discarded. Global average pooling and two independent four-class heads are optimized jointly with cross-entropy on the original images.

## 3. Numerical semantic reasoning

The original training algorithms are intentionally retained under `Generative_model/` as requested.

```bash
python Generative_model/Train_tokenizer/Step1-generate_train_data_CNs_LCs.py
python Generative_model/Train_tokenizer/Step2-train_CNs_LCs.py
python Generative_model/Stage1_generative_pretraining_CNs_LCs.py --cfg Generative_model/config/config_CNs_LCs_grading.json
python Generative_model/Stage2_finetune_CNs_LCs.py \
  --train_json data/fold1/text_train.json \
  --val_json data/fold1/text_val.json \
  --num_epochs 200 \
  --batch_size 64 \
  --learning_rate 1e-4
```

The retained scripts contain dataset-specific default paths from the initial release. Pass explicit paths or update the JSON configuration before training. The paper's text setting is numerical-aware BPE, T5, 200 epochs, AdamW, batch size 64, and learning rate `1e-4`.

## 4. Bi-directional multimodal fusion

```bash
python -m scripts.train_fusion \
  --manifest data/fold1/fusion_train.json \
  --image-root data \
  --vision-checkpoint checkpoints/fold1/vision_grader.pt \
  --text-model models/t5-clinical-base \
  --text-checkpoint checkpoints/fold1/text_grader.pt \
  --tokenizer checkpoints/numerical_tokenizer \
  --output checkpoints/fold1/fusion.pt
```

The script freezes both adapted encoders. It trains only the projections, shared queries, cross-attention/calibration layers, and two task-specific classifiers using AdamW, batch size 64, and learning rate `1e-4`. Local masked and global visual representations are concatenated as visual tokens. The calibrated visual and textual query tokens are concatenated, pooled, and sent to both classifiers.

## Evaluation

Prediction JSON must contain the two labels and two four-element probability vectors. Validate the interface with the synthetic example:

```bash
python -m scripts.evaluate_predictions examples/predictions.example.json
```

The output contains overall ACC, four within-level correct-classification rates, four one-vs-rest AUCs, and macro-AUC. Run every experiment independently on five folds and report mean plus or minus standard deviation. The paper used two-sided Wilcoxon signed-rank tests with Benjamini-Hochberg correction for comparisons.

## Verification

```bash
python -m compileall -q biominer scripts Generative_model tests
python -m unittest discover -s tests -v
```

The tests verify topology masks and losses, text JSON rendering, metric definitions, output shapes, and the mandatory freezing of both encoders during fusion.

## Reproducibility boundary

The repository can reproduce the algorithm and input contracts, but the removed participant data, segmentation masks, trained weights, exact fold manifests, the locally deployed Qwen3 corpus-generation environment, and one unreported Gaussian-noise standard deviation are necessary to reproduce the paper's numerical tables exactly. Supply those original artifacts without changing participant-level folds to perform a result-level reproduction.
