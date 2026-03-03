"""
SME-YOLO: Small-target Multi-scale Enhanced YOLO.

Based on YOLOv11n via Ultralytics.
The paper (arxiv 2601.11402) introduces NWD loss, EUCB upsampling, and MSFA
attention on top of YOLOv11n.  No public code release exists yet, so this
module uses a standard YOLOv11n as the base.  Custom NWD / EUCB / MSFA layers
are noted below as future extension points.

Install:  pip install ultralytics
"""

from ultralytics import YOLO


def build(num_classes, pretrained=True, variant='yolo11n', **kwargs):
    """
    Load YOLOv11n (base for SME-YOLO).

    Args:
        num_classes: detection classes (background excluded).
        pretrained:  if True, load COCO-pretrained weights.
        variant:     'yolo11n' (nano) recommended per paper.

    Returns:
        ultralytics.YOLO object.
    """
    weight_file = f'{variant}.pt' if pretrained else f'{variant}.yaml'
    model = YOLO(weight_file)
    print(f'[sme_yolo] Loaded {weight_file}  (pretrained={pretrained})')
    # TODO: future — inject custom NWD loss, EUCB, MSFA modules here
    return model


def train_sme_yolo(model, data_yaml, epochs=20, imgsz=640, batch=16, **kwargs):
    """Train with Ultralytics trainer."""
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        **kwargs,
    )
    return results


def evaluate_sme_yolo(model, data_yaml, **kwargs):
    """Run validation."""
    return model.val(data=data_yaml, **kwargs)
