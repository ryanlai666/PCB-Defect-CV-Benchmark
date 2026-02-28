"""
Dataset, transforms, and DataLoader factories for DeepPCB defect detection.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

from config import BATCH_SIZE, TEST_SPLIT, RANDOM_SEED


# ─── Dataset ────────────────────────────────────────────────────────────────

class PCBDefectDataset(Dataset):
    """
    Reads DeepPCB-format data: each line in the manifest has
    '<image_path> <annotation_path>'.  Annotation files contain rows of
    'x1 y1 x2 y2 class_id'.
    """

    def __init__(self, manifest_lines, data_dir='', transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.samples = []

        for line in manifest_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                img_path, ann_path = parts
                self.samples.append((img_path.replace('.jpg', '_test.jpg'), ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, ann_rel_path = self.samples[idx]

        img_path = os.path.join(self.data_dir, img_rel_path)
        ann_path = os.path.join(self.data_dir, ann_rel_path)

        # Load image
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        boxes, labels = [], []

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2 = map(float, parts[:4])
                        class_id = int(parts[4])
                        x_min, x_max = min(x1, x2), max(x1, x2)
                        y_min, y_max = min(y1, y2), max(y1, y2)
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': tv_tensors.BoundingBoxes(
                boxes_tensor,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            'labels': labels_tensor,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # ── Strip tv_tensors wrappers ──────────────────────────────────
        # v2 transforms return tv_tensors.Image and tv_tensors.BoundingBoxes.
        # Faster R-CNN's GeneralizedRCNNTransform does its own resizing and
        # the __torch_function__ override on tv_tensors types causes reshape
        # errors (e.g. "shape '[640,640]' invalid for size 1228800").
        # Converting to plain torch.Tensor avoids this entirely.
        if hasattr(image, 'as_subclass'):
            image = image.as_subclass(torch.Tensor)
        if hasattr(target['boxes'], 'as_subclass'):
            target['boxes'] = target['boxes'].as_subclass(torch.Tensor)

        return image, target


# ─── Transforms ─────────────────────────────────────────────────────────────

def get_v2_transforms(train=True):
    """Torchvision v2 transform pipeline with automatic bbox handling."""
    if train:
        return v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ToDtype(torch.float32, scale=True),
            v2.SanitizeBoundingBoxes(),
        ])
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])


# ─── DataLoader helpers ─────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate for variable-size bbox targets (Faster R-CNN style)."""
    return tuple(zip(*batch))


def create_dataloaders(data_dir, batch_size=BATCH_SIZE):
    """
    Reads trainval.txt / test.txt from *data_dir*, splits trainval into
    train/val, and returns three DataLoaders.
    """
    with open(os.path.join(data_dir, 'trainval.txt'), 'r') as f:
        trainval_lines = f.readlines()
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        test_lines = f.readlines()

    train_lines, val_lines = train_test_split(
        trainval_lines, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
    )

    train_dataset = PCBDefectDataset(train_lines, data_dir=data_dir,
                                     transforms=get_v2_transforms(train=True))
    val_dataset   = PCBDefectDataset(val_lines,   data_dir=data_dir,
                                     transforms=get_v2_transforms(train=False))
    test_dataset  = PCBDefectDataset(test_lines,  data_dir=data_dir,
                                     transforms=get_v2_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
