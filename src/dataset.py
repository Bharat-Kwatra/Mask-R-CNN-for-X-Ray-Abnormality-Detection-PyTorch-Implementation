import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import rle2mask, masks_to_boxes
from .config import IMAGE_SIZE

class XRayDataset(Dataset):
    """
    Custom PyTorch Dataset for the SIIM-ACR Pneumothorax Segmentation dataset.
    """
    def __init__(self, df, image_dir, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame with ImageId and EncodedPixels.
            image_dir (str): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        
        # Group annotations by ImageId
        self.image_ids = df['ImageId'].unique()
        self.image_annotations = {img_id: df[df['ImageId'] == img_id] for img_id in self.image_ids}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load Image
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        
        # Load masks and create bounding boxes
        annotation_df = self.image_annotations[image_id]
        rle_strings = annotation_df['EncodedPixels'].values
        
        # Handle cases with no pneumothorax (all RLEs are -1)
        if len(rle_strings) == 1 and rle_strings[0].strip() == '-1':
            # No objects, return empty tensors
            num_objs = 0
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Filter out the '-1' RLEs for images that have positive masks
            positive_rles = [rle for rle in rle_strings if rle.strip() != '-1']
            num_objs = len(positive_rles)
            
            # Decode RLEs to binary masks
            # Note: The original image size was 1024x1024
            raw_masks = [rle2mask(rle, 1024, 1024) for rle in positive_rles]
            
            # The masks must be resized to the new image size
            # We convert to PIL Image to use the resize function
            pil_masks = [Image.fromarray(mask) for mask in raw_masks]
            resized_masks = [m.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST) for m in pil_masks]
            masks = np.array([np.array(m) for m in resized_masks])

            # Get bounding boxes from masks
            boxes = masks_to_boxes(masks)
            
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            # All objects are "Pneumothorax", so label is 1
            labels = torch.ones((num_objs,), dtype=torch.int64)

        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        
        # Apply transforms if any
        if self.transforms:
            image = self.transforms(image)

        return image, target

def get_transform():
    """Defines the image transformations for training."""
    custom_transforms = [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(custom_transforms)

def collate_fn(batch):
    """
    Custom collate function for the DataLoader, since the model expects a list of
    images and a list of targets.
    """
    return tuple(zip(*batch))
