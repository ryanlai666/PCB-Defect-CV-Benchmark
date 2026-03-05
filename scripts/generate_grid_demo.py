import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

models = [
    ('faster_rcnn', 'Faster R-CNN'),
    ('vit_det', 'ViT-Det'),
    ('sme_yolo', 'SME-YOLO'),
    ('yolo26', 'YOLO26'),
    ('rt_detr', 'RT-DETR'),
    ('deimv2_l', 'DEIMv2-L')
]

# just get indices from sme_yolo
sme_files = glob.glob("results/demo/sme_yolo/sme_yolo_*.jpg")
sme_files.sort()
indices = []
for f in sme_files:
    basename = os.path.basename(f)
    idx = basename.split('_')[-1].split('.')[0]
    indices.append(idx)

# select a few distinct ones
if len(indices) >= 60:
    selected_indices = [indices[5], indices[25], indices[55]]
elif len(indices) >= 3:
    selected_indices = [indices[0], indices[len(indices)//2], indices[-1]]
else:
    selected_indices = indices

for idx in selected_indices:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Defect Detection Comparison (Image {idx})', fontsize=20)
    
    for i, (model_dir, model_name) in enumerate(models):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        img_path = os.path.join('results/demo', model_dir, f"{model_dir}_{idx}.jpg")
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(model_name, fontsize=16)
        else:
            ax.set_title(f"{model_name} (Missing)", fontsize=16)
            
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    out_path = os.path.join('results', f'grid_demo_comparison_{idx}.jpg')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

print("Done generating grids.")
