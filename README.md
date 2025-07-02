# transfer-learning-mobilenetv2

## Overview

**transfer-learning-mobilenetv2** is a Python project dedicated to exploring and implementing accelerated transfer learning techniques using [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) with the powerful MobileNetV2 model. This repository is designed for students, researchers, and practitioners who want to leverage state-of-the-art pre-trained models for fast and efficient feature extraction and classification on custom image datasets.

Transfer learning enables you to take advantage of rich feature representations learned from large-scale datasets (such as ImageNet) and apply them to your own domain — significantly reducing training time and improving accuracy, even with limited data.

---

## Features

- **MobileNetV2 Backbone:** Uses TensorFlow/Keras’s MobileNetV2 as a feature extractor, with options for fine-tuning.
- **Accelerated Transfer Learning:** Implement both standard and accelerated transfer learning pipelines.
- **Custom Training Scripts:** Modular code for loading data, splitting datasets, and model training.
- **Metrics & Evaluation:** Compute and visualize confusion matrices, precision, recall, and F1 scores for model performance.
- **K-Fold Cross Validation:** Easily test the stability and generalization of your model.
- **No Proprietary Dependencies:** Uses only open-source Python libraries.
- **Easy to Extend:** Well-commented and modular code for experimenting with different architectures or datasets.

---

## Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.x and Keras
- NumPy
- Matplotlib, Seaborn (optional, for plotting)
- Your dataset (organized in subfolders by class, e.g., `./small_flower_dataset`)

Install dependencies:
```bash
pip install tensorflow keras numpy matplotlib seaborn
```

### Usage

1. **Clone the Repository**
    ```bash
    git clone https://github.com/avro1199/transfer-learning-mobilenetv2.git
    cd transfer-learning-mobilenetv2
    ```

2. **Prepare Your Dataset**
   - Place your images in subfolders under a root directory (e.g., `small_flower_dataset/rose`, `small_flower_dataset/sunflower`).

3. **Train with Transfer Learning**
    ```bash
    python TransferLearning.py
    ```

   - Customize hyperparameters, dataset path, or training options at the top of the script or via arguments.

4. **View Results**
   - The script will display training and validation accuracy/loss, confusion matrix, and classwise precision/recall/F1 scores.
   - Extend the code for advanced plotting, model saving, or feature extraction as needed.

---

## Repository Structure

- `TransferLearning.py`      — Main implementation with all key functions and training/testing pipelines.
- `TransferLearning_Rj.py`   — Alternative or experimental implementation.
- `README.md`                — This file.
- `LICENSE`                  — GNU General Public License v3.0.
- `small_flower_dataset/`    — Example dataset structure (not included; add your own images).
- `requirements.txt`         — (Optional) List of dependencies.

---

## Example: Transfer Learning Workflow

```python
from keras.applications import MobileNetV2
from keras import layers, Model

# Load base model without top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add new classifier layers
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## License

This project is licensed under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE).

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or additional features.

---

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for their robust machine learning ecosystem.
- MobileNetV2 authors for their efficient architecture.
- Academic modules and open source communities for foundational concepts and code inspiration.
