<div align="center"> 
<h1 align="center">Understanding cell fate decisions with temporal attention</h1>

<a href="https://doi.org/10.1371/journal.pcbi.1014813">
  <img src="https://img.shields.io/badge/Paper-PLOS%20Comp%20Bio-red?style=flat&label=Paper&link=https://doi.org/10.1371/journal.pcbi.1014813" alt="Static Badge" />
</a>
<a href="https://repository.publisso.de/resource/frl:6526608">
  <img src="https://img.shields.io/badge/Dataset-blue?style=flat&label=Data&link=https://repository.publisso.de/resource/frl:6526608"
  alt="Static Badge" />
</a>
<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Static Badge" />
</a>
</div>

---

## Introduction

<p align="center">
  <img alt="Teaser Image" src="./assets/overview.svg">
  <br>
    <em>Overview of the proposed cell fate prediction framework and the explainability using temporal attention analysis.</em>
</p>

Code for [**Understanding cell fate decisions with temporal attention**](https://doi.org/10.1371/journal.pcbi.1014813) (PLOS Computational Biology).

We predict whether a cell will undergo apoptosis or mitosis from single-cell time-lapse microscopy sequences. A ResNet-50 backbone encodes each frame, and a Transformer encoder with attention pooling aggregates the sequence into a fate prediction. We then use the temporal attention weights to identify the frames the model relies on and relate them to morphological features and fluorescent reporter dynamics (p53, cell cycle, circadian) through per-sequence statistical tests.

This repository contains the extraction of single-cell sequences from the raw data, the training and evaluation pipeline, the statistical analysis scripts, and the trained model checkpoint.

## Repository structure

```
├── environment.yml
├── requirements.txt
└── src
    ├── train.py              # training
    ├── test.py               # evaluation, incl. truncation experiments
    ├── analysis.py           # export attention weights and per-frame cell features
    ├── configs/              # training, backbone and transformer configs
    ├── dataset/              # dataset and dataloader
    ├── models/               # ResNet-50 backbone and Transformer
    ├── utils/                # config factory, annotation parser, transforms, cell features
    ├── data
    │   ├── dataset/          # extracted sequences go here (sequences/)
    │   ├── analysis/         # analysis outputs
    │   └── scripts/          # sequence extraction, statistics, figure
    └── output
        └── model/            # trained model: configs, checkpoint, evaluation results
```

## Installation

The model checkpoint and larger files are stored with [Git LFS](https://git-lfs.com), so install Git LFS before cloning.

```bash
git lfs install
git clone https://github.com/bozeklab/Cell-Fate-Prediction.git
cd Cell-Fate-Prediction
conda env create -f environment.yml -n cell-fate
conda activate cell-fate
```


Alternatively, install the dependencies with `pip install -r requirements.txt`.

All commands below are run from the repository root.

## Data

1. Download the raw time-lapse dataset (image sequences and annotations) from [here](https://doi.org/10.4126/FRL01-006526608).

2. Extract the single-cell sequences:

```bash
python src/data/scripts/create_sequences.py \
    --data_dir /path/to/raw_data \
    --ann_file <annotation_file>.json \
    --output_dir src/data/dataset/sequences
```

`--ann_file` is relative to `--data_dir`. The script crops a 112×112 patch (`--patch_size`) around each tracked cell in every frame and keeps only cells that either die or divide. It saves one `.pt` file per cell, grouped by video, and writes a stratified 80/10/10 train/validation/test split with a fixed seed:

```
src/data/dataset/sequences/
├── <video_name>/
│   ├── <cell_id>.pt
│   └── ...
├── train_cells.csv
├── val_cells.csv
└── test_cells.csv
```

## Trained model

`src/output/model/` contains the model used in the paper:

- `configs/`: the configurations used for training
- `training/checkpoints/model_checkpoint.safetensors`: the trained weights
- `evaluation/`: the evaluation results reported in the paper

The default arguments of `test.py` and `analysis.py` point to this model.

## Training

```bash
python src/train.py --name my_model
```

Hyperparameters are set in `src/configs/training.yml`, `backbone.yml` and `transformer.yml`. Checkpoints and a copy of the configs are written to `<output_dir>/<name>/training`, where `output_dir` is defined in `training.yml`. Pick a new name so the provided model in `src/output/model/` is not overwritten. After training, the best checkpoint is evaluated on the test set.

Training logs to the Weights & Biases project `cell_fate_prediction` by default; pass `--no-log` to disable logging.

## Evaluation
 
```bash
python src/test.py
```
 
This evaluates the provided model on the test split and writes `<output_dir>/<experiment_name>/results.csv`, by default `src/output/model/evaluation/overall/results.csv`. The file contains F1, accuracy, macro precision and recall, loss, and per-class precision and recall.
 
<details>
<summary><b>Arguments</b></summary>

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `src/output/model` | Model folder containing `configs/` |
| `--model_checkpoint` | `src/output/model/training/checkpoints/model_checkpoint.safetensors` | Checkpoint to evaluate |
| `--output_dir` | `src/output/model/evaluation` | Folder for the evaluation results |
| `--experiment_name` | `overall` | Subfolder of `--output_dir` for this run |
| `--truncation` | `0` | Number of frames N for the truncation experiments (see below) |
| `--use_truncation` | off | Keep only the last N frames instead of discarding them |
| `--debug` | off | Run the dataloader in debug mode |
 
To evaluate your own model, set `--model_path` and `--model_checkpoint` to it, and set `--output_dir` so its results are not written into the provided model's folder.

</details>

### Truncation experiments

```bash
# Discard the last N frames before the event (predict N frames ahead)
python src/test.py --truncation 20 --experiment_name discard_last_20

# Use only the last N frames
python src/test.py --truncation 20 --use_truncation --experiment_name last_20_only
```

Results are written to `src/output/model/evaluation/<experiment_name>/results.csv`, so use a distinct experiment name for each run.

## Attention and feature analysis

**1. Export attention weights and cell features**

```bash
python src/analysis.py --dataset_path src/data/dataset/sequences --cell_analysis
```

This runs the model on each sequence individually and writes `src/data/analysis/cell_analysis.csv`. For every sequence, the file contains:

- the true and predicted label and the dose
- the attention weight of each frame
- per-frame morphology: area, circularity, eccentricity, equivalent diameter, perimeter, solidity and number of neighbors
- the mean intensity of the three reporter channels

<details>
<summary><b>Arguments</b></summary>

| Argument | Default | Description |
|---|---|---|
| `--dataset_path` | required | Folder with the extracted sequences |
| `--model_path` | `src/output/model` | Model folder containing `configs/` |
| `--model_checkpoint` | `src/output/model/training/checkpoints/model_checkpoint.safetensors` | Checkpoint to analyze |
| `--cell_analysis` | off | Add the per-frame cell features; required for steps 2 and 3 |
| `--truncation` | none | Discard the last N frames; the output is written to `cell_analysis_trunc_<N>.csv` |

</details>

**2. Statistical tests**

```bash
python src/data/scripts/compute_statistics.py
```

The script uses correctly classified sequences of at least 50 frames. Within each sequence, it splits the frames into high-attention frames (at or above the 90th percentile of that sequence's attention) and the remaining frames. Each feature is compared between the two groups with a permutation test (50,000 permutations), and the script reports Cliff's δ and Cohen's d (not used for further analysis) as effect sizes.

Results are written to `src/data/analysis/statistics/<Death|Division>_<feature>.csv`, one row per sequence. The quantile, minimum sequence length and number of permutations are constants in the script's `__main__` block. The script uses all CPU cores and takes a while to run.

## Citation

```bibtex
@article{10.1371/journal.pcbi.1014813,
    doi = {10.1371/journal.pcbi.1014813},
    author = {Bürger, Florian AND Dias Gomes, Martim AND Granada, Adrián E. AND Moreau, Noémie AND Bozek, Katarzyna},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {Understanding cell fate decisions with temporal attention},
    year = {2026},
    month = {10},
    volume = {22},
    url = {https://doi.org/10.1371/journal.pcbi.1014813},
    pages = {1-20},
    number = {10},
}
```

## Dataset

The dataset is available on [PUBLISSO](https://repository.publisso.de/resource/frl:6526608) and was introduced in [TransientTrack](https://arxiv.org/abs/2512.01885) (accepted at IEEE Transactions on Medical Imaging). If you use the data, please cite both the dataset and the paper:


```bibtex
@misc{bürger2025transienttrackadvancedmultiobjecttracking,
      title={TransientTrack: Advanced Multi-Object Tracking and Classification of Cancer Cells with Transient Fluorescent Signals}, 
      author={Florian Bürger and Martim Dias Gomes and Nica Gutu and Adrián E. Granada and Noémie Moreau and Katarzyna Bozek},
      year={2025},
      eprint={2512.01885},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01885}, 
}
```
```bibtex
@misc{buerger2025dataset,
  author    = {Bürger, Florian and Dias Gomes, Martim and Gutu, Nica and Granada, Adrián E. and Moreau, Noémie and Bozek, Katarzyna},
  title     = {Large-Scale Time-Lapse Microscopy Dataset for Cell Tracking},
  publisher = {PUBLISSO},
  year      = {2025},
  doi       = {10.4126/FRL01-006526608},
  url       = {https://repository.publisso.de/resource/frl:6526608},
  note      = {Data set, CC BY 4.0}
}
```

## License

The code and the trained model weights in this repository are released under the [MIT License](LICENSE). The dataset is published separately on PUBLISSO under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).