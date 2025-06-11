import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    PROCESSED_TRAIN_CSV_PATH, PROCESSED_TRAIN_IMAGE_DIR,
    DEVICE, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS,
    LEARNING_RATE, RANDOM_SEED, VALIDATION_SPLIT, OUTPUT_DIR
)
from model import get_model_instance
from dataset import XRayDataset, get_transform, collate_fn

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")

    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())

    return total_loss / len(data_loader)

def validate_one_epoch(model, data_loader, device):
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {0}/{NUM_EPOCHS} [Validation]")

    with torch.no_grad():
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # In evaluation mode, the model returns predictions, but if targets are provided,
            # it also calculates the losses in training mode internally. So we switch to train mode
            # for the forward pass to get validation loss, without tracking gradients.
            model.train()
            loss_dict = model(images, targets)
            model.eval() # Switch back to eval mode
            
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            progress_bar.set_postfix(val_loss=losses.item())
            
    return total_loss / len(data_loader)


def main():
    """Main function to run the training pipeline."""
    print(f"Using device: {DEVICE}")

    # Load preprocessed annotations
    if not os.path.exists(PROCESSED_TRAIN_CSV_PATH):
        print(f"Error: Processed annotation file not found at {PROCESSED_TRAIN_CSV_PATH}")
        print("Please run preprocess.py first.")
        return
        
    df = pd.read_csv(PROCESSED_TRAIN_CSV_PATH)
    
    # We only want to train on images that have masks
    df_positive = df[df['EncodedPixels'].str.strip() != '-1'].copy()
    image_ids = df_positive['ImageId'].unique()
    
    # Split data
    train_ids, val_ids = train_test_split(
        image_ids, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
    )
    
    train_df = df_positive[df_positive['ImageId'].isin(train_ids)]
    val_df = df_positive[df_positive['ImageId'].isin(val_ids)]

    print(f"Training on {len(train_ids)} images, validating on {len(val_ids)} images.")

    # Create datasets
    train_dataset = XRayDataset(df=train_df, image_dir=PROCESSED_TRAIN_IMAGE_DIR, transforms=get_transform())
    val_dataset = XRayDataset(df=val_df, image_dir=PROCESSED_TRAIN_IMAGE_DIR, transforms=get_transform())

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )

    # Initialize model
    model = get_model_instance(NUM_CLASSES)
    model.to(DEVICE)

    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss = validate_one_epoch(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} (Val loss: {best_val_loss:.4f})")
            
        # Update learning rate
        # lr_scheduler.step()

    print("Training finished.")

if __name__ == '__main__':
    main()
