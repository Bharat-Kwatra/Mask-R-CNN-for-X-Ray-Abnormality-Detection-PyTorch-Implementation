import os
import torch

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Original Data
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "dicom-images-train")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "dicom-images-test")
TRAIN_RLE_PATH = os.path.join(DATA_DIR, "train-rle.csv")

# Processed Data (after running preprocess.py)
PROCESSED_TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "processed_images_train")
PROCESSED_TRAIN_CSV_PATH = os.path.join(OUTPUT_DIR, "processed_train_annotations.csv")

# --- MODEL AND TRAINING PARAMS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512 # Images will be resized to this dimension
BATCH_SIZE = 4
NUM_WORKERS = 2 # For DataLoader
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
VALIDATION_SPLIT = 0.1 # 10% of data for validation
RANDOM_SEED = 42

# There is only one class: "Pneumothorax" + background
NUM_CLASSES = 2 

# --- INFERENCE PARAMS ---
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
PREDICTION_THRESHOLD = 0.5 # Confidence score threshold for predictions

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
