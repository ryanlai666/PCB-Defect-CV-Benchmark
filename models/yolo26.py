"""
YOLO26 via the Ultralytics package.

Install:  pip install ultralytics
Docs:     https://docs.ultralytics.com/models/yolo26/

Training uses Ultralytics' built-in trainer (NOT PyTorch DataLoader).
"""

from ultralytics import YOLO


def build(num_classes, pretrained=True, variant='yolo26n', **kwargs):
    """
    Load a YOLO26 model.

    Args:
        num_classes: number of detection classes (background excluded).
        pretrained:  if True, load COCO-pretrained weights.
        variant:     model size — 'yolo26n' (nano) | 'yolo26s' | 'yolo26m' | 'yolo26l' | 'yolo26x'

    Returns:
        ultralytics.YOLO object.
        Train with:  model.train(data='data.yaml', epochs=N, imgsz=640)
    """
    weight_file = f'{variant}.pt' if pretrained else f'{variant}.yaml'
    model = YOLO(weight_file)
    print(f'[yolo26] Loaded {weight_file}  (pretrained={pretrained})')
    return model


def train_yolo26(model, data_yaml, epochs=20, imgsz=640, batch=16, **kwargs):
    """Convenience wrapper around model.train()."""
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        **kwargs,
    )
    return results


def evaluate_yolo26(model, data_yaml, **kwargs):
    """Run validation."""
    return model.val(data=data_yaml, **kwargs)
