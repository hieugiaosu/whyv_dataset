```markdown
# Target Speaker Extraction (TSE) Dataset Repository

This repository provides PyTorch-compatible datasets and recipes for evaluating target speaker extraction systems. It includes:

- A **Vietnamese Self Evaluation Dataset** for evaluating target speaker extraction using Vietnamese data.
- A recipe for creating a **Mandarin TSE Dataset (AISHELL3-2Mix)** based on the AISHELL3 dataset, combined with WHAM noise data.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
  - [Vietnamese Self Evaluation Dataset](#vietnamese-self-evaluation-dataset)
  - [Mandarin TSE Dataset Recipe (AISHELL3-2Mix)](#mandarin-tse-dataset-recipe-aishell3-2mix)
- [Installation and Requirements](#installation-and-requirements)
  - [Using the Vietnamese Dataset](#using-the-vietnamese-dataset)
  - [Preparing and Using the Mandarin Dataset](#preparing-and-using-the-mandarin-dataset)
- [Repository Structure](#repository-structure)
- [Acknowledgments](#acknowledgments)

---

## Overview

This repository is designed to support research on target speaker extraction (TSE) by providing:

- A ready-to-use Vietnamese evaluation dataset.
- A recipe and supporting code to generate a Mandarin TSE dataset (AISHELL3-2Mix) by combining the AISHELL3 corpus with WHAM noise.

Both datasets are implemented in PyTorch and are easy to integrate into your projects.

---

## Datasets

### Vietnamese Self Evaluation Dataset

The Vietnamese dataset is intended for evaluating target speaker extraction. It provides:
- **Mixture audio files**: Containing overlapping speech.
- **Ground truth audio files**: Isolated speech for comparison.
- **Speaker embeddings**: Reference embeddings for target speakers.


#### Example Code

You can import and use the Vietnamese dataset as follows:

```python
from vietnamese_dataset import VietnameseSelfEvaluationDataset

# Download and instantiate the dataset
dataset = VietnameseSelfEvaluationDataset.download(root="./", sampling_rate=8000)
```

---

### Mandarin TSE Dataset Recipe (AISHELL3-2Mix)

This recipe details how to create a Mandarin target speaker extraction dataset by combining:
- **AISHELL3 Dataset**: A Mandarin speech corpus. More details at [AISHELL3](https://openslr.org/93/).
- **WHAM Noise Dataset**: Background noise for realistic mixtures. Download it via:
  - **Direct download using curl**:
    ```bash
    curl -O https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
    ```
  - Or visit [WHAM](http://wham.whisper.ai/).

#### Preparation Script

A preparation function is provided to generate the AISHELL3-2Mix dataset:

```python
from AISHELL3_2Mix import prepare_aishell_mix_dataset

prepare_aishell_mix_dataset(
    aishell_3_folder_path,
    wham_noise_folder_path,
    output_folder_path,
)
```

Make sure to download and extract both the AISHELL3 and WHAM datasets before running the script.

#### Using the Mandarin Dataset

After preparing the dataset, load it using the provided PyTorch dataset class:

```python
from AISHELL3_2Mix import AiShell3mixDataset

ds = AiShell3mixDataset(
    path="path_to_aishell_3_mix",
    mode="noisy",  # Options: "noisy" or "clean"
    sample_rate=8000,
    segment_length=4
)
```

This class allows you to choose between noisy mixtures or clean signals and customize the sample rate and segment length to suit your evaluation needs.

---

## Installation and Requirements

Ensure you have Python 3.7+ and install the required packages:

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

---

## Repository Structure

```
├── README.md
├── vietnamese_dataset
│   ├── __init__.py
│   └── dataset.py        
├── AISHELL3_2Mix                
│   └── __init__.py  
└── requirements.txt
```


## Acknowledgments

- **AISHELL3 Dataset:** [https://openslr.org/93/](https://openslr.org/93/)
- **WHAM Noise Dataset:** [http://wham.whisper.ai/](http://wham.whisper.ai/)
- This repository is dataset for the paper: [demo page](https://anonymous.4open.science/w/whyv/)