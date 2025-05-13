# Food101 Image Classification with EfficientNetB0

This project is an assignment task which demonstrates image classification on the Food101 dataset using TensorFlow and transfer learning with EfficientNetB0.

## Overview

The goal of this project was to build a deep learning model capable of classifying images into 101 different food categories using the Food101 dataset. Due to time and resource constraints, the model was only partially trained, resulting in lower-than-expected accuracy. Consequently, it was not integrated into an application.

## Features

- TensorFlow Datasets integration for loading Food101
- Image preprocessing and augmentation
- Transfer learning using EfficientNetB0
- Two-stage training: initial training with frozen base model, then fine-tuning
- Early stopping and model checkpointing

## Dataset

- **Source:** [Food101](https://www.tensorflow.org/datasets/catalog/food101)
- **Classes:** 101 food categories
- **Size:** 75,750 training images, 25,250 validation images

## Model Architecture

- **Base Model:** EfficientNetB0 (pretrained on ImageNet)
- **Top Layers:** GlobalAveragePooling, Dropout, Dense with softmax
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam

## Limitations

- The model did not reach high accuracy due to limited training time and computational constraints.
- Full fine-tuning was not completed, and integration into a deployment application was not pursued due to these limitations.

## How to Run

1. Install TensorFlow and TensorFlow Datasets:
   ```bash
   pip install tensorflow tensorflow-datasets
   ```

2. Run the notebook:
   ```bash
   jupyter notebook Food101_FineTuning.ipynb
   ```

3. The notebook will:
   - Load and preprocess the data
   - Train a model in two phases
   - Save the best model as `best_model.h5`

## Future Improvements

- Apply advanced learning rate scheduling
- Finetune more layers of EfficientNetB0
- Use mixed precision training for efficiency
- Integrate the trained model into a mobile/web app

## Acknowledgements

- TensorFlow and TensorFlow Datasets
- Food101 Dataset by Lukas Bossard et al.
