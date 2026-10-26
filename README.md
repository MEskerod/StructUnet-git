# StructUnet

StructUnet is a U-Net-based model for RNA secondary structure prediction.

## Abstract
**Background**: RNA secondary structure prediction remains a central challenge in computational biology. While deep-learning methods have improved predictive accuracy, many approaches retain constraints inherited from traditional thermodynamic models, including restrictions on pseudoknots and non-canonical base pairs. Here we present \textbf{StructUnet}, a U‑Net-based convolutional neural network that combines an RNA input representation with post‑processing that permits both pseudoknots and non‑canonical interactions. StructUnet was trained on RNAStrAlign and evaluated on a held-out, non-overlapping RNAStrAlign test set and the analogous ArchiveII benchmark.    
**Results**: Analysis of the RNAStrAlign dataset showed that approximately 5\% of the annotated base pairs were non-canonical, attributable to the substantial representation of 16S RNA sequences. Allowing these non-canonical base pairs, along with pseudoknots, during decoding improved predictive performance relative to restricted decoding. StructUnet achieved F1‑scores of \textbf{0.96} on RNAStrAlign and \textbf{0.85} on ArchiveII, outperforming ViennaRNA, CONTRAfold, and CNNfold under the adopted benchmark settings. StructUnet predicted non-canonical interactions with an F1-score of 0.92 and achieved higher pseudoknot-specific performance than CNNfold while maintaining a low pseudoknot-presence error. Performance remained high for RNA families represented in the training data but decreased for unseen families, highlighting limitations in dataset diversity and model generalisation.      
**Conclusion**: StructUnet provides evidence that RNA secondary structure prediction can benefit from relaxing historical modelling assumptions that exclude interaction types represented in comparative RNA secondary-structure annotations. By enabling prediction of pseudoknots and non-canonical base pairs while maintaining strong benchmark performance, StructUnet broadens the structural scope of deep learning-based RNA secondary structure prediction and highlights the need for more diverse structural datasets.} 

## Installation

Run the following from the repository root:

```bash
conda env create -f StructUnet.yml
conda activate StructUnet
```

[StructUnet.yml](StructUnet.yml) includes PyTorch, CUDA 12.1, and `gwf`. It targets NVIDIA GPU systems and does not require editing an installation path.

## Data and models

Download datasets and pretrained checkpoints from the [shared data and model folder](https://drive.google.com/drive/folders/15VAdY8AYT4Z6OosgDE6-HZ-c1UZ5YQeW?usp=sharing). Place datasets in `data/` and checkpoints in `models/`.

The current pipeline converts **TSV tables to HDF5 datasets**. Models support 8- or 16-channel input; use the same `--input_dim` for conversion, training, and prediction.

The examples use `models/StructUnet8_dataA.pth`. Pass your checkpoint with `-m`; otherwise the prediction scripts look for `models/RNA_Unet8.pth` or `models/RNA_Unet16.pth`.

## Usage

Run commands from the repository root. Use `--help` for additional options.

### Predict from a sequence or FASTA file

[scripts/predict.py](scripts/predict.py) accepts either a sequence string or a FASTA file:

```bash
mkdir -p results
python scripts/predict.py -i input.fasta -o results/prediction.bpseq -m models/StructUnet8_dataA.pth --input_dim 8
```

Replace `input.fasta` with a sequence such as `GGGAAACCC` to predict directly. Output supports `.bpseq` and `.ct`; multiple FASTA records produce separate numbered files. Omit `-o` to print structures to the terminal.

### Prepare a dataset

[scripts/split_data.py](scripts/split_data.py) prepares TSV tables and train/validation/test labels from source CSVs and an existing cluster file. If you already have prepared TSVs, start with conversion below.

The TSV must contain `id`, `sequence`, `pairs`, and `family`. `pairs` is a Python-list literal of zero-based partner indices, with `None` for unpaired bases. A split column such as `split_A` contains `train`, `val`, or `test` labels.

```bash
TMPDIR="$(mktemp -d)" 
python scripts/convert_data.py data/RNAStrAlign.tsv data/converted/RNAStrAlign8_dataA --input_dim 8 --split split_A --cores 4 --plot
```

This creates `RNAStrAlign8_dataA_train.h5`, `RNAStrAlign8_dataA_val.h5`, and `RNAStrAlign8_dataA_test.h5` in `data/converted/`. Omit `--split` to create one HDF5 file for the whole dataset. The output prefix should not include `.h5`.

Use a fresh writable `TMPDIR` for each conversion; temporary files remain there after completion. `--plot` saves dataset distribution plots to `figures/`.

### Train a model

[scripts/training.py](scripts/training.py) takes training and validation HDF5 files followed by a model name:

```bash
mkdir -p models
python scripts/training.py data/converted/RNAStrAlign8_dataA_train.h5 data/converted/RNAStrAlign8_dataA_val.h5 StructUnet8_dataA --input_dim 8 --epochs 150 --patience 10
```

The best checkpoint is saved to `models/StructUnet8_dataA.pth`; logs, history, and plots go to `steps/training_StructUnet8_dataA/`. Use a new model name for an independent run, since existing checkpoints and histories are loaded when present.

### Predict or evaluate a dataset

To save predictions without computing metrics:

```bash
python scripts/predict_dataset.py data/converted/RNAStrAlign8_dataA_test.h5 -o results/predictions.h5 -m models/StructUnet8_dataA.pth --input_dim 8
```

To predict and evaluate against the reference structures:

```bash
python scripts/evaluate_dataset.py data/converted/RNAStrAlign8_dataA_test.h5 -o StructUnet8_dataA_evaluation -m models/StructUnet8_dataA.pth --input_dim 8
```

For evaluation, `-o` is a name prefix, not a directory. Predictions and per-sequence, overall, and family-level metric tables go to `results/`; the F1 plot goes to `figures/`. Metrics include precision, recall, F1, shift-tolerant scores, and pseudoknot metrics.

Both dataset scripts accept `--device auto`, `cpu`, or `cuda`. Their default input dimension is 16, so specify `--input_dim 8` when using an 8-channel model.

### Post-processing

The default is `--postprocessing unrestricted`.

| Mode | Pairing mask | Pseudoknots |
| --- | --- | --- |
| `unrestricted` | None | Allowed |
| `restricted` | Standard pairs and minimum separation | Allowed |
| `unrestricted_noncrossing` | None | Excluded |
| `restricted_noncrossing` | Standard pairs and minimum separation | Excluded |

The restricted mask permits AU, CG, and GU pairs in either direction, as well as pairs involving N, with separation greater than three positions. Single-sequence prediction supports only `restricted` and `unrestricted`; the dataset scripts support all four modes.

To compare modes, first save raw predictions with `--postprocessing None`:

```bash
python scripts/predict_dataset.py data/converted/RNAStrAlign8_dataA_test.h5 -o results/raw_predictions.h5 -m models/StructUnet8_dataA.pth --input_dim 8 --postprocessing None
python scripts/compare_postprocessing.py results/raw_predictions.h5 data/converted/RNAStrAlign8_dataA_test.h5 -o postprocessing_comparison --cores 4
```

This adds decoded matrices to the prediction file and saves comparison tables and plots under `results/postprocessing_comparison/` and `figures/postprocessing_comparison/`, including non-canonical-pair and pseudoknot metrics.

## Additional tools

| Script | Purpose |
| --- | --- |
| [evaluate_and_plot.py](scripts/evaluate_and_plot.py) | Compare saved predictions against a shared reference dataset |
| [plot_combined_evaluation.py](scripts/plot_combined_evaluation.py) | Compare F1 distributions from evaluation CSVs |
| [analysis.py](scripts/analysis.py) | Generate publication analyses and figures from experiment outputs |

## Cluster workflow

[workflow.py](workflow.py) uses `gwf` to connect preparation, training, evaluation, and comparison jobs. Adjust its account, GPU queue, Conda environment, CUDA setup, and input paths for your cluster before running it.

```bash
gwf status
gwf run train_StructUnet8_dataA
gwf run evaluate_model_8_dataA
```

Targets may also schedule upstream dependencies. The optional ViennaRNA, CONTRAfold, and CNNfold targets reference scripts that are not included in this checkout. The current preprocessing template needs its `echo` quote corrected, and the conversion template's declared family-plot filename needs to match the converter output before relying on workflow completion tracking.

## Citation
