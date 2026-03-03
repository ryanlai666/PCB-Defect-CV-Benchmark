"""
Faster R-CNN with custom CNN backbone (original notebook model).
Also supports a pretrained ResNet-50 FPN backbone.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator


class CustomCNNBackbone(nn.Module):
    """Lightweight 5-layer CNN feature extractor."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(),
        )
        self.out_channels = 256

    def forward(self, x):
        return self.features(x)


def _create_custom_fasterrcnn(num_classes):
    """Original notebook model: custom CNN backbone + Faster R-CNN head."""
    backbone = CustomCNNBackbone()

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2,
    )
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )


def _create_pretrained_fasterrcnn(num_classes):
    """Faster R-CNN v2 with ResNet-50 FPN pretrained on COCO."""
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # Replace the classification head to match our num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes,
        )
    )
    return model


def build(num_classes, pretrained=True, **kwargs):
    """Entry point called by the model registry."""
    if pretrained:
        print('[faster_rcnn] Using pretrained ResNet-50 FPN v2 backbone (COCO)')
        return _create_pretrained_fasterrcnn(num_classes)
    else:
        print('[faster_rcnn] Using custom CNN backbone (no pretrained weights)')
        return _create_custom_fasterrcnn(num_classes)
