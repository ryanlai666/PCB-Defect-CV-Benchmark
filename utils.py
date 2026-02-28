"""
Utility functions — mainly YOLO-format conversion for Ultralytics-based models.
"""

import os
import shutil
import yaml
from PIL import Image

from config import CLASS_MAP, NUM_CLASSES


def convert_deeppcb_to_yolo(data_dir, output_dir):
    """
    Convert DeepPCB annotations to YOLO format.

    DeepPCB format (per .txt):  x1 y1 x2 y2 class_id
    YOLO format   (per .txt):  class_id cx cy w h      (all normalised 0-1)

    Creates the directory structure expected by Ultralytics:
        output_dir/
          images/
            train/  val/  test/
          labels/
            train/  val/  test/
          data.yaml
    """
    from sklearn.model_selection import train_test_split
    from config import TEST_SPLIT, RANDOM_SEED

    # Read manifest
    with open(os.path.join(data_dir, 'trainval.txt'), 'r') as f:
        trainval_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        test_lines = [l.strip() for l in f if l.strip()]

    train_lines, val_lines = train_test_split(
        trainval_lines, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
    )

    splits = {'train': train_lines, 'val': val_lines, 'test': test_lines}

    for split_name, lines in splits.items():
        img_out = os.path.join(output_dir, 'images', split_name)
        lbl_out = os.path.join(output_dir, 'labels', split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                continue
            img_rel, ann_rel = parts
            img_rel = img_rel.replace('.jpg', '_test.jpg')

            img_src = os.path.join(data_dir, img_rel)
            ann_src = os.path.join(data_dir, ann_rel)

            if not os.path.exists(img_src):
                continue

            # Get image dimensions
            with Image.open(img_src) as im:
                w, h = im.size

            # Copy image
            img_basename = os.path.basename(img_rel)
            shutil.copy2(img_src, os.path.join(img_out, img_basename))

            # Convert annotation
            lbl_basename = os.path.splitext(img_basename)[0] + '.txt'
            yolo_lines = []
            if os.path.exists(ann_src):
                with open(ann_src, 'r') as af:
                    for aline in af:
                        aline = aline.strip()
                        if not aline:
                            continue
                        aparts = aline.split()
                        if len(aparts) < 5:
                            continue
                        x1, y1, x2, y2 = map(float, aparts[:4])
                        class_id = int(aparts[4])

                        # YOLO class IDs are 0-indexed
                        yolo_cls = class_id - 1

                        # Normalise to 0-1
                        cx = ((x1 + x2) / 2.0) / w
                        cy = ((y1 + y2) / 2.0) / h
                        bw = abs(x2 - x1) / w
                        bh = abs(y2 - y1) / h
                        yolo_lines.append(f'{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

            with open(os.path.join(lbl_out, lbl_basename), 'w') as lf:
                lf.write('\n'.join(yolo_lines))

    # Write data.yaml
    create_yolo_yaml(output_dir)
    print(f'YOLO dataset created at: {output_dir}')


def create_yolo_yaml(output_dir, num_classes=NUM_CLASSES - 1,
                     class_names=None):
    """
    Write a data.yaml for Ultralytics trainers.
    NOTE: YOLO num_classes does NOT include background, so we subtract 1.
    """
    if class_names is None:
        # 0-indexed names list
        class_names = [CLASS_MAP[i + 1] for i in range(num_classes)]

    yaml_path = os.path.join(output_dir, 'data.yaml')
    data = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val':   'images/val',
        'test':  'images/test',
        'nc':    num_classes,
        'names': class_names,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f'Created {yaml_path}')
    return yaml_path
