# Pneumothorax Segmentation using PyTorch Mask R-CNN


## Overview

This project provides a robust, end-to-end pipeline for detecting and segmenting instances of pneumothorax in chest X-ray images using a **Mask R-CNN** model. The entire workflow is built on modern, industry-standard libraries, primarily **PyTorch** and **TorchVision**, ensuring high performance and maintainability.

The codebase is intentionally structured into a series of modular Python scripts, offering a clean and scalable alternative to monolithic Jupyter notebooks. This approach promotes code reusability, simplifies debugging, and makes the entire process—from data preparation to model inference—transparent and easy to follow.

---

## Project Structure

The repository is organized to clearly separate concerns, with distinct directories for data, source code, and outputs.

```
.
├── data/
│   ├── dicom-images-train/  # Raw DICOM training images (user-provided).
│   └── train-rle.csv        # Original annotations file (user-provided).
├── output/
│   ├── processed_images_train/ # PNG images created by preprocess.py.
│   ├── processed_train_annotations.csv # Cleaned annotation manifest.
│   └── best_model.pth       # Saved model weights with the best validation score.
├── src/
│   ├── config.py          # Centralized configuration for all parameters and paths.
│   ├── dataset.py         # PyTorch Dataset class for loading and transforming X-ray data.
│   ├── model.py           # Defines the Mask R-CNN architecture using TorchVision.
│   ├── predict.py         # Script for running inference on new images with a trained model.
│   ├── train.py           # The main script to execute the model training and validation loops.
│   ├── utils.py           # Core helper functions (e.g., RLE decoding, box calculation).
│   └── visualize.py       # Utilities to draw predictions (masks, boxes) on images.
├── preprocess.py          # Standalone script for initial data preparation.
└── requirements.txt       # All required Python packages for project setup.
```
---

## Setup and Installation

Follow these steps to set up your local environment.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd pneumothorax-detection-pytorch
    ```

2.  **Create a Virtual Environment (Recommended):**
    This practice isolates the project's dependencies and avoids conflicts with other projects.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    This command installs all necessary packages listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download and Place Data:**
    This project requires the dataset from the **SIIM-ACR Pneumothorax Segmentation Challenge**. Download the data and place the following items into the `data/` directory:
    * `dicom-images-train/` (folder)
    * `train-rle.csv` (file)

---

## End-to-End Workflow

The pipeline is designed to be executed in three distinct stages.

### Step 1: Preprocess Raw Data

**Script:** `preprocess.py`

This is a crucial first step that standardizes the dataset for training.
* **What it does:** It reads the original medical-grade DICOM images, converts them to a standard 3-channel PNG format, and resizes them to a uniform dimension (e.g., 512x512 pixels) suitable for the model.
* **Inputs:** `data/dicom-images-train/` and `data/train-rle.csv`.
* **Outputs:** Creates `output/processed_images_train/` with the new PNGs and `output/processed_train_annotations.csv`, a clean manifest file linking images to their annotations.

**To Run:**
```bash
python preprocess.py
```

### Step 2: Train the Mask R-CNN Model

**Script:** `src/train.py`

This script is the core of the project, handling model training and validation.
* **What it does:** It loads the processed data using the custom `XRayDataset` class, which handles on-the-fly mask decoding. It then initializes a Mask R-CNN model with a pre-trained ResNet-50 backbone (transfer learning) and fine-tunes it on the X-ray data. The script automatically saves the model that achieves the lowest validation loss.
* **Inputs:** The processed images and CSV file in the `output/` directory.
* **Output:** The best-performing model weights, saved as `output/best_model.pth`.

**To Run:**
```bash
python src/train.py
```
*Note: All hyperparameters (learning rate, batch size, epochs, etc.) can be easily adjusted in `src/config.py`.*

### Step 3: Run Inference and Visualize Predictions

**Script:** `src/predict.py`

After training, this script uses your model to make predictions on any image.
* **What it does:** It loads the saved `best_model.pth` weights. Given the path to an image, it performs the necessary transformations, feeds the image to the model, and retrieves the predicted segmentation masks, bounding boxes, and confidence scores.
* **Inputs:** A path to an image file and the path to the trained model (`--model_path`).
* **Output:** A new image file saved in the `output/` directory, with the model's predictions visually overlaid.

**To Run:**
```bash
python src/predict.py --image_path /path/to/your/image.png
```

---
## Core Component Deep Dive

* `src/config.py`: A centralized file holding all constants and configurable parameters. This includes file paths, image dimensions, device settings (CPU/GPU), and training hyperparameters. Modifying this file is the easiest way to experiment with different settings.

* `src/dataset.py`: Defines the `XRayDataset` class, the bridge between your data on disk and the PyTorch model. Its key responsibility is to load an image and its corresponding RLE strings, convert the RLE to a binary mask tensor, compute bounding boxes, and format everything into the target dictionary required by TorchVision's Mask R-CNN.

* `src/model.py`: Contains a single function, `get_model_instance()`, which constructs the Mask R-CNN model. It loads a pre-trained model from TorchVision and replaces the final prediction heads (for both boxes and masks) with new, untrained layers that match the number of classes in our specific problem (1 for Pneumothorax + 1 for background).

* `src/utils.py`: A collection of essential helper functions. The most important is `rle2mask`, which efficiently decodes the run-length encoding strings from the dataset into 2D NumPy arrays representing binary masks.

* `src/visualize.py`: Provides the `visualize_predictions` function, which takes an image and the model's output to produce a human-readable visualization. It draws semi-transparent masks, colored bounding boxes, and labels with confidence scores on the image.

---
## Citation & Acknowledgements

This project is built upon the dataset and problem statement from the **SIIM-ACR Pneumothorax Segmentation Challenge**.

* **Dataset Source**: [Kaggle Competition Page](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)
* The original repository this work is based on can be found here: [wenqinglee/maskrcnn-xray](https://github.com/wenqinglee/maskrcnn-xray)

This is a sample script and workflow adapted on works of **Dr. Bharat Kwatra**.

## Disclaimer

This project and its associated code are provided for educational and research purposes only. The trained model is a proof-of-concept and is **not intended for clinical diagnosis, medical advice, or any real-world patient care.**
