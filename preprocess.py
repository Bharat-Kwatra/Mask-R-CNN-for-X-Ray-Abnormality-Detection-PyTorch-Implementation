import os
import pydicom
import pandas as pd
from PIL import Image
from tqdm import tqdm
import sys

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from config import (
    TRAIN_IMAGE_DIR, TEST_IMAGE_DIR, TRAIN_RLE_PATH,
    PROCESSED_TRAIN_IMAGE_DIR, PROCESSED_TRAIN_CSV_PATH,
    IMAGE_SIZE
)

def rle2mask(rle, width, height):
    """
    Converts a run-length encoding to a binary mask.
    This is a utility function that might be needed, but for preprocessing,
    we are mainly focused on image conversion. The dataset loader will handle masks.
    """
    import numpy as np
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T

def preprocess_data():
    """
    Reads original DICOM images, resizes them, saves them as PNGs,
    and creates a new annotation CSV file pointing to these new images.
    """
    print("Starting data preprocessing...")

    # Ensure output directories exist
    os.makedirs(PROCESSED_TRAIN_IMAGE_DIR, exist_ok=True)

    # Load annotations
    df = pd.read_csv(TRAIN_RLE_PATH)
    print(f"Loaded {len(df)} annotations.")

    # Filter out images with no pneumothorax finding
    # We will still process all images, but create a new dataframe for training
    df_with_masks = df[df[' EncodedPixels'].str.strip() != '-1']
    image_ids_with_masks = df_with_masks['ImageId'].unique()
    print(f"Found {len(image_ids_with_masks)} images with positive masks.")

    processed_records = []
    
    # Get all unique image IDs from the RLE file
    all_image_ids = df['ImageId'].unique()
    
    print(f"Processing {len(all_image_ids)} unique images...")

    for image_id in tqdm(all_image_ids, desc="Processing Images"):
        dicom_path = os.path.join(TRAIN_IMAGE_DIR, f"{image_id}.dcm")
        
        if not os.path.exists(dicom_path):
            print(f"Warning: DICOM file not found for {image_id}. Skipping.")
            continue

        # Read DICOM and convert to PIL Image
        dicom_dataset = pydicom.dcmread(dicom_path)
        image = Image.fromarray(dicom_dataset.pixel_array)
        image = image.convert("RGB") # Model expects 3 channels

        # Resize image
        resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        # Save as PNG
        png_path = os.path.join(PROCESSED_TRAIN_IMAGE_DIR, f"{image_id}.png")
        resized_image.save(png_path)
        
        # Get all annotations for this image
        annotations = df[df['ImageId'] == image_id][' EncodedPixels'].tolist()
        
        # We need to add one record per image, with a list of its masks
        # For this script, we just create a mapping. The dataset loader will group them.
        for rle in annotations:
             processed_records.append({
                'ImageId': image_id,
                'EncodedPixels': rle,
                'ImagePath': png_path
            })

    # Create and save the new dataframe
    processed_df = pd.DataFrame(processed_records)
    processed_df.to_csv(PROCESSED_TRAIN_CSV_PATH, index=False)
    
    print("-" * 30)
    print("Preprocessing complete!")
    print(f"Saved {len(all_image_ids)} processed images to: {PROCESSED_TRAIN_IMAGE_DIR}")
    print(f"Saved new annotation file to: {PROCESSED_TRAIN_CSV_PATH}")
    print("-" * 30)


if __name__ == '__main__':
    preprocess_data()
