# Sound-Detection-Model

## Overview

The **AIISH Sound Detection Model** repository contains the machine learning framework for environmental sound classification developed as part of the AIISH project.

The repository implements a complete pipeline for environmental sound recognition, including audio preprocessing, feature extraction, model training, evaluation, and inference. It follows a transfer learning approach using **YAMNet** as the pretrained audio feature extractor, followed by a custom neural network classifier for environmental sound classification.

The framework has been designed with a modular architecture, enabling efficient experimentation with different datasets, model architectures, and training strategies while supporting deployment on resource-constrained devices.

---

## Repository Contents

This repository includes:

- Audio preprocessing utilities
- Feature extraction using YAMNet
- Environmental sound classification models
- Transfer learning pipeline
- Model training and evaluation framework
- Performance metrics and benchmarking
- Trained model artifacts
- Testing and validation utilities

---

## Repository Structure

```text
Sound-Detection-Model/
│
├── checkpoints/            # Training checkpoints
├── data/                   # Dataset storage
├── models/                 # Trained model artifacts
├── notebooks/              # Development notebooks
├── results/                # Evaluation outputs
├── saved_models/           # Exported model formats
│
├── src/
│   ├── data/               # Dataset loading and preprocessing
│   ├── evaluation/         # Evaluation metrics and analysis
│   ├── model/              # Model architecture and YAMNet interface
│   ├── training/           # Training and feature extraction pipeline
│   └── utils/              # Utility functions
│
├── tests/                  # Testing and validation scripts
├── yamnet_official/        # Official YAMNet implementation
│
├── requirements.txt
├── audioset_labels.txt
└── README.md
```

---

## Model Architecture

The current implementation follows a transfer learning pipeline for environmental sound classification.

```text
Environmental Audio
        │
        ▼
Audio Preprocessing
        │
        ▼
Pretrained YAMNet
        │
        ▼
Audio Embedding Extraction
        │
        ▼
Custom AIISH Classification Network
        │
        ▼
Environmental Sound Classification
```

The modular design enables future improvements such as backbone fine-tuning, custom feature extractors, and alternative classifier architectures without requiring significant modifications to the overall pipeline.

---

## Technologies

The implementation is built using:

- Python
- TensorFlow
- TensorFlow Hub
- Keras
- NumPy
- Pandas
- Librosa
- Scikit-learn

---

## Datasets

The model development utilizes the following datasets:

### AudioSet

AudioSet is a large-scale collection of human-labeled environmental audio events developed by Google. The pretrained **YAMNet** model is trained on AudioSet and provides generalized audio representations that serve as the foundation for transfer learning within this project.

### UrbanSound8K

UrbanSound8K is a benchmark dataset consisting of 8,732 labeled audio clips spanning ten urban environmental sound classes. It is used for supervised training, validation, and benchmarking of the environmental sound classification model.

### Custom AIISH Environmental Sound Dataset

The custom AIISH dataset is being developed specifically for this project and consists of real-world recordings of environmental sounds relevant to assistive hearing applications. It includes critical, household, communication, transportation, and public environmental sounds that are either underrepresented or absent in existing public datasets.

---

## Repository Components

| Directory | Description |
|-----------|-------------|
| `src/data` | Dataset loading and preprocessing utilities |
| `src/model` | Model architecture and pretrained model interfaces |
| `src/training` | Feature extraction and model training pipelines |
| `src/evaluation` | Model evaluation, metrics, and label mapping |
| `src/utils` | Shared utility functions |
| `tests` | Testing and validation scripts |
| `models` | Trained model artifacts |
| `results` | Evaluation reports and generated outputs |
| `yamnet_official` | Official YAMNet implementation |

---

## Development Workflow

The repository supports the complete model development lifecycle, including:

- Audio preprocessing
- Feature extraction
- Transfer learning
- Model training
- Model evaluation
- Model inference
- Performance benchmarking

The modular implementation enables reproducible experimentation while maintaining a clear separation between data processing, model development, training, and evaluation.

---

## License

This repository is intended for academic research and educational purposes. Licensing information may be updated as the project evolves.
