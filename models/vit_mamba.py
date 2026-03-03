"""
ViT-Mamba hybrid model using MambaVision (NVlabs/MambaVision).

MambaVision fuses Mamba state-space layers with Transformer self-attention
in a hierarchical vision backbone.  We wrap it as a Faster R-CNN backbone
for PCB defect detection, using its multi-scale intermediate features.

Install:  pip install mambavision timm
Repo:     https://github.com/NVlabs/MambaVision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from config import IMG_SIZE


class MambaVisionBackbone(nn.Module):
    """
    Loads a MambaVision model from the mambavision package and exposes
    the last-stage feature map for Faster R-CNN.
    """

    def __init__(self, model_name='mambavision_t_1k', pretrained=True,
                 img_size=IMG_SIZE):
        super().__init__()
        from mambavision import create_model as mv_create_model

        self.model = mv_create_model(model_name, pretrained=pretrained)
        self.img_size = img_size

        # MambaVision models have hierarchical stages like ConvNeXt/Swin.
        # Typical dims for mambavision_t: [80, 160, 320, 640]
        # We detect the actual dim by a dummy forward pass.
        self._out_channels = None
        self._detect_out_channels()

        # 1x1 conv to unify channel count for RPN
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(self._out_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.out_channels = 256

    @torch.no_grad()
    def _detect_out_channels(self):
        """Infer the output channel dim with a dummy input."""
        dummy = torch.zeros(1, 3, 224, 224)
        try:
            feats = self.model.forward_features(dummy)
            if feats.dim() == 4:
                self._out_channels = feats.shape[1]
            elif feats.dim() == 3:
                self._out_channels = feats.shape[-1]
            else:
                self._out_channels = feats.shape[-1]
        except Exception:
            self._out_channels = getattr(self.model, 'num_features', 640)

    def forward(self, x):
        # Safety resize: MambaVision is hierarchical-conv-based so it
        # handles variable sizes, but we normalise for consistency.
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        feats = self.model.forward_features(x)

        if feats.dim() == 3:
            # (B, N, D) → reshape to 2D spatial map
            B, N, D = feats.shape
            h = w = int(N ** 0.5)
            feats = feats.permute(0, 2, 1).reshape(B, D, h, w)

        # feats: (B, C, H, W)
        feats = self.channel_reduce(feats)
        return feats


def build(num_classes, pretrained=True,
          model_name='mambavision_t_1k', img_size=IMG_SIZE, **kwargs):
    """
    Create MambaVision + Faster R-CNN detector.

    Available model_name variants:
        'mambavision_t_1k'   — tiny  (ImageNet-1K pretrained)
        'mambavision_t2_1k'  — tiny2
        'mambavision_s_1k'   — small
        'mambavision_b_1k'   — base
        'mambavision_l_1k'   — large
        'mambavision_l2_1k'  — large2

    Args:
        num_classes: total classes including background.
        pretrained:  load ImageNet-pretrained MambaVision weights.
        img_size:    input resolution for Faster R-CNN transform.
    """
    backbone = MambaVisionBackbone(model_name=model_name, pretrained=pretrained,
                                   img_size=img_size)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Match Faster R-CNN's internal resize to what the backbone expects
        min_size=img_size,
        max_size=img_size,
    )

    tag = 'ImageNet pretrained' if pretrained else 'random init'
    print(f'[vit_mamba] MambaVision ({model_name}) + Faster R-CNN  ({tag})')
    return model
