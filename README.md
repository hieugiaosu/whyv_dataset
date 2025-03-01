# WHYV Evaluate Dataset Repository

This repository provides PyTorch-compatible datasets and recipes for evaluating **Target Speaker Extraction (TSE)** systems. It includes:

- A **Vietnamese Self-Evaluation Dataset** for testing TSE systems with Vietnamese speech data.
- A recipe for creating a **Mandarin TSE Dataset (AISHELL3-2Mix)** using the AISHELL-3 corpus combined with WHAM noise data.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
  - [Vietnamese Self-Evaluation Dataset](#vietnamese-self-evaluation-dataset)
  - [Mandarin TSE Dataset (AISHELL3-2Mix)](#mandarin-tse-dataset-aishell3-2mix)
- [Installation and Requirements](#installation-and-requirements)
  - [Using the Vietnamese Dataset](#using-the-vietnamese-dataset)
  - [Preparing and Using the Mandarin Dataset](#preparing-and-using-the-mandarin-dataset)
- [Repository Structure](#repository-structure)
- [Acknowledgments](#acknowledgments)

---

## Overview

This repository supports research in **Target Speaker Extraction (TSE)** by offering:

- A pre-built **Vietnamese Self-Evaluation Dataset** for immediate use.
- A recipe and code to generate a **Mandarin TSE Dataset (AISHELL3-2Mix)** by mixing the AISHELL-3 Mandarin speech corpus with WHAM noise.

Both datasets are implemented in PyTorch, making them easy to integrate into your TSE projects.

---

## Datasets

### Vietnamese Self-Evaluation Dataset

This dataset is designed for evaluating TSE systems using Vietnamese speech. It includes:

- **Mixture Audio Files**: Overlapping speech recordings.
- **Ground Truth Audio Files**: Isolated target speaker audio for comparison.
- **Speaker Embeddings**: Precomputed embeddings for identifying target speakers.

#### Example Usage

To use the Vietnamese dataset, download and instantiate it as follows:

```python
from vietnamese_dataset import VietnameseSelfEvaluationDataset

# Download and load the dataset
dataset = VietnameseSelfEvaluationDataset.download(root="./data", sampling_rate=8000)
```

---

### Mandarin TSE Dataset (AISHELL3-2Mix)

This section provides a recipe to create a Mandarin TSE dataset by combining:

- **AISHELL-3 Dataset**: A Mandarin speech corpus available at [OpenSLR](https://openslr.org/93/).
- **WHAM Noise Dataset**: Background noise for realistic audio mixtures, downloadable from [WHAM](http://wham.whisper.ai/) or via:
  ```bash
  curl -O https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
  ```

#### Preparation Steps

Use the provided script to generate the AISHELL3-2Mix dataset:

```python
from AISHELL3_2Mix import prepare_aishell_mix_dataset

prepare_aishell_mix_dataset(
    aishell_3_folder_path="path/to/aishell3",
    wham_noise_folder_path="path/to/wham_noise",
    output_folder_path="path/to/output",
)
```

**Note**: Ensure both AISHELL-3 and WHAM datasets are downloaded and extracted before running the script.

#### Loading the Dataset

Once prepared, load the dataset with the provided PyTorch class:

```python
from AISHELL3_2Mix import AiShell3mixDataset

dataset = AiShell3mixDataset(
    path="path/to/aishell_3_mix",
    mode="noisy",  # Options: "noisy" or "clean"
    sample_rate=8000,
    segment_length=4,  # Segment length in seconds
)
```

Customize the `mode`, `sample_rate`, and `segment_length` to fit your evaluation needs.

---

## Installation and Requirements

### Prerequisites
- Python 3.7 or higher
- PyTorch and Torchaudio

### Setup
Install the required dependencies:

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

---

## Repository Structure

```
├── README.md               # This file
├── vietnamese_dataset/     # Vietnamese dataset module
│   ├── __init__.py
│   └── dataset.py         # Dataset implementation
├── AISHELL3_2Mix/         # Mandarin dataset module
│   ├── __init__.py
│   └── [other_files]      # Recipe and dataset code
└── requirements.txt        # Python dependencies
```

---

## Acknowledgments

- **AISHELL-3 Dataset**: Available at [https://openslr.org/93/](https://openslr.org/93/)
- **WHAM Noise Dataset**: Available at [http://wham.whisper.ai/](http://wham.whisper.ai/)
- This repository supports the dataset for the paper: [Demo Page](https://anonymous.4open.science/w/whyv/)
