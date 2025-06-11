import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance(num_classes):
    """
    Creates an instance of the Mask R-CNN model with a pre-trained backbone.

    Args:
        num_classes (int): The number of classes, including the background.

    Returns:
        torch.nn.Module: The Mask R-CNN model.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT")

    # --- Replace the box predictor ---
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # --- Replace the mask predictor ---
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model
