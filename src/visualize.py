import numpy as np
import cv2
import random

# Define a color for the masks and boxes
INSTANCE_CATEGORY_NAMES = ['__background__', 'Pneumothorax']
PNEUMOTHORAX_COLOR = (0, 0, 255) # Red in BGR format for OpenCV

def get_random_color():
    """Generates a random BGR color."""
    return [random.randint(0, 255) for _ in range(3)]

def visualize_predictions(image, predictions, threshold=0.5):
    """
    Draws masks, bounding boxes, and labels on an image.

    Args:
        image (np.ndarray): The input image in RGB format.
        predictions (dict): The model's output, containing 'boxes', 'labels', 'scores', 'masks'.
        threshold (float): The score threshold to display a prediction.

    Returns:
        np.ndarray: The image with visualizations, in BGR format for OpenCV.
    """
    # Convert RGB image to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_with_masks = img_bgr.copy()

    scores = predictions['scores'].cpu().numpy()
    
    high_conf_indices = np.where(scores > threshold)[0]
    
    if not len(high_conf_indices):
        print("No predictions above threshold found.")
        return img_bgr

    boxes = predictions['boxes'][high_conf_indices].cpu().numpy()
    labels = predictions['labels'][high_conf_indices].cpu().numpy()
    # Masks are [N, 1, H, W], so we squeeze the channel dimension
    masks = predictions['masks'][high_conf_indices].cpu().numpy().squeeze(1)

    for i in range(len(boxes)):
        # --- Draw Mask ---
        mask = masks[i]
        # Create a solid color image for the mask
        color_mask = np.zeros_like(img_bgr)
        color_mask[mask > 0.5] = PNEUMOTHORAX_COLOR
        # Blend the mask with the image
        img_with_masks = cv2.addWeighted(img_with_masks, 1.0, color_mask, 0.5, 0)
        
        # --- Draw Bounding Box ---
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(img_with_masks, (x1, y1), (x2, y2), PNEUMOTHORAX_COLOR, 2)

        # --- Draw Label ---
        score = scores[high_conf_indices[i]]
        label_text = f"{INSTANCE_CATEGORY_NAMES[labels[i]]}: {score:.2f}"
        
        # Position the text
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(img_with_masks, (x1, text_y - text_height - baseline), (x1 + text_width, text_y), PNEUMOTHORAX_COLOR, -1)
        cv2.putText(img_with_masks, label_text, (x1, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_with_masks
