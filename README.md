# StructUnet

StructUnet is a U-Net-based model for RNA secondary structure prediction.

## Abstract
**Background**: RNA secondary structure prediction remains a central challenge in computational biology. While deep-learning methods have improved predictive accuracy, many approaches retain constraints inherited from traditional thermodynamic models, including restrictions on pseudoknots and non-canonical base pairs. Here we present **StructUnet**, a U‑Net-based convolutional neural network that combines an RNA input representation with post‑processing that permits both pseudoknots and non‑canonical interactions. StructUnet was trained on RNAStrAlign and evaluated on the held-out RNAStrAlign test set and the external ArchiveII benchmark.    
**Results**: Analysis of RNAStrAlign showed that approximately 5\% of annotated base pairs were non-canonical and that these interactions occured in nearly one-third of sequences. Allowing non-canonical base pairs and pseudoknots during decoding improved predictive performance relative to restricted decoding. StructUnet achieved F1‑scores of **0.98** on RNAStrAlign and **0.91** on ArchiveII, outperforming ViennaRNA, CONTRAfold, CNNfold, and Ufold on both benchmarks. StructUnet predicted non-canonical interactions with an F1-score of 0.95 and achieved higher pseudoknot-specific performance than CNNfold while maintaining a low pseudoknot-presence error. Performance remained high for RNA families represented in the training data but decreased for unseen families, highlighting limitations in dataset diversity and model generalisation.    
**Conclusion**: StructUnet provides evidence that RNA secondary structure prediction can benefit from relaxing historical modelling assumptions that exclude biologically relevant structural features. By enabling prediction of pseudoknots and non-canonical base pairs while maintaining strong benchmark performance, StructUnet broadens the structural scope of deep learning-based RNA secondary structure prediction and highlights the need for more diverse structural datasets.

`TODO`: Add link to publication.  

## Data and Models
Pretrained models and larger dataset files can be found at [here](https://drive.google.com/drive/folders/15VAdY8AYT4Z6OosgDE6-HZ-c1UZ5YQeW?usp=sharing)

## Overview

This repository contains code for:
- predicting RNA secondary structures from a sequence or FASTA file
- converting raw RNA structure datasets into the repository input format
- training StructUnet on converted datasets
- predicting and evaluating an entire dataset

The main user-facing scripts are in [`scripts/`](/Users/au579798/Documents/StructUnet-git/scripts).

## Repository structure

```text
StructUnet-git/
├── README.md
├── StructUnet.yml
├── scripts/
│   ├── predict.py
│   ├── convert_data.py
│   ├── training.py
│   ├── predict_dataset.py
│   ├── evaluate_dataset.py
│   └── utils/
├── data/
└── results/
```

## Installation

Create the Conda environment from [`StructUnet.yml`](/Users/au579798/Documents/StructUnet-git/StructUnet.yml):

```bash
conda env create -f StructUnet.yml
conda activate StructUnet
```

If the environment file contains a fixed local prefix, update it before creating the environment.

## Data format

Converted datasets are stored as:
- one pickle file per sequence
- pickle file lists such as `train.pkl`, `valid.pkl`, `test.pkl`, or `archive.pkl`

Each per-sequence pickle stores:

```python
namedtuple("RNA", "input output length family name sequence")
```

where:
- `input` is the input tensor representation
- `output` is the target contact map
- `length` is sequence length
- `family` is the RNA family label
- `name` is the source file name
- `sequence` is the RNA sequence

## Models

By default, the scripts expect pretrained checkpoints in the repository root:
- `models/RNA_Unet.pth` for 8-channel input
- `models/RNA_Unet16.pth` for 16-channel input

Custom checkpoints can be supplied with `-m/--model`.

## Usage

### 1. Predict from sequence or FASTA

Use [`scripts/predict.py`](/Users/au579798/Documents/StructUnet-git/scripts/predict.py) to predict one or more structures from a sequence string or FASTA file.

Example:

```bash
python scripts/predict.py -i input.fasta -o prediction.bpseq --input_dim 8 --postprocessing unrestricted
```

Notes:
- valid output formats are `.bpseq` and `.ct`
- `--postprocessing restricted` uses the masked post-processing step
- `--postprocessing unrestricted` uses the permissive decoding path
- `-std/--only_standardbp` is currently kept as a compatibility alias for restricted post-processing

### 2. Convert a raw dataset

Use [`scripts/convert_data.py`](/Users/au579798/Documents/StructUnet-git/scripts/convert_data.py) to convert a raw `.tar.gz` archive of `.ct` files into the repository format.

Example:

```bash
python scripts/convert_data.py input.tar.gz converted_data --input_dim 8 --RNAStralign --split --data_distribution
```

This can:
- convert each structure into a separate pickle file
- split the dataset into training, validation, and test sets
- write `train.pkl`, `valid.pkl`, and `test.pkl`
- optionally plot family and length distributions

### 3. Train a model

Use [`scripts/training.py`](/Users/au579798/Documents/StructUnet-git/scripts/training.py) to train StructUnet on a converted dataset.

Example:

```bash
python scripts/training.py converted_data model.pth --input_dim 8 --hidden_dim 64 --epochs 150 -lr 0.01 -wd 0.0 -p 5
```

Input requirements:
- the data directory must contain `train.pkl` and `valid.pkl`

Training outputs:
- model checkpoint
- `training_log/`
- training history CSV
- loss and F1 plots

### 4. Predict an entire dataset

Use [`scripts/predict_dataset.py`](/Users/au579798/Documents/StructUnet-git/scripts/predict_dataset.py) to run inference on every sample in a dataset file list.

Example:

```bash
python scripts/predict_dataset.py converted_data/test.pkl -o results/predictions -m models/RNA_Unet8.pth --input_dim 8 --postprocessing unrestricted
```

This writes one prediction pickle per sequence to the requested output directory.

### 5. Predict and evaluate an entire dataset

Use [`scripts/evaluate_dataset.py`](/Users/au579798/Documents/StructUnet-git/scripts/evaluate_dataset.py) to predict every sample in a dataset and compute evaluation metrics.

Example:

```bash
python scripts/evaluate_dataset.py converted_data/test.pkl -o results/evaluation -m models/RNA_Unet8.pth --input_dim 8 --postprocessing unrestricted
```

This script:
- predicts each sample in the dataset
- saves one prediction pickle per sequence
- writes one row per sequence to `per_sequence_metrics.csv`
- writes aggregate metrics to `summary_metrics.csv`
- writes family-level metrics to `family_metrics.csv`
- writes `per_sequence_F1.png` if plotting succeeds

The default evaluation currently includes:
- precision
- recall
- F1
- precision, recall, and F1 with one-basepair shift allowed
- weighted F1
- pseudoknot pair metrics
- pseudoknot sequence-level metrics
- false pseudoknot rate

Non-canonical evaluation is intentionally not part of `evaluate_dataset.py`.

## Outputs

The [`results/`](/Users/au579798/Documents/StructUnet-git/results) directory is the default home for evaluation outputs.

Typical contents include:
- `predictions/`
- `per_sequence_metrics.csv`
- `summary_metrics.csv`
- `family_metrics.csv`
- `per_sequence_F1.png`


## Citation

`TODO`: Add citation information after publication.
