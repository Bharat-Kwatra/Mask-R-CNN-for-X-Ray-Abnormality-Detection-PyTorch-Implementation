import numpy as np

def rle2mask(rle_string, width, height):
    """
    Converts a run-length encoding string to a binary mask.

    Args:
        rle_string (str): The RLE string. Can be '-1' for empty masks.
        width (int): The width of the original image.
        height (int): The height of the original image.

    Returns:
        np.ndarray: A 2D numpy array representing the binary mask.
    """
    if rle_string.strip() == '-1':
        return np.zeros((height, width), dtype=np.uint8)

    rows, cols = height, width
    rle_numbers = [int(num) for num in rle_string.split()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    
    img = np.zeros(rows * cols, dtype=np.uint8)
    
    for index, (start, length) in enumerate(rle_pairs):
        # RLE starts from 1, Python from 0
        start_index = start - 1
        end_index = start_index + length
        img[start_index:end_index] = 1
        
    img = img.reshape(cols, rows)
    img = img.T # RLE is column-major
    
    return img

def masks_to_boxes(masks):
    """
    Computes bounding boxes from binary masks.

    Args:
        masks (np.ndarray): A 3D numpy array of shape [N, H, W], where N is the number of masks.

    Returns:
        np.ndarray: A 2D numpy array of shape [N, 4] with bounding boxes in (x1, y1, x2, y2) format.
    """
    if masks.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    n = masks.shape[0]
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        y, x = np.where(masks[i])
        if len(x) > 0 and len(y) > 0:
            boxes[i, 0] = np.min(x)
            boxes[i, 1] = np.min(y)
            boxes[i, 2] = np.max(x)
            boxes[i, 3] = np.max(y)
    return boxes
