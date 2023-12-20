# Using V3Det to detect objects in images, run on the original training dataset of v3d

# This script is adapted from

import os
import torch

from pathlib import Path
from mmdet.apis import DetInferencer
from IPython.utils import io
from tqdm import tqdm

# Set working directory
wdir = Path('/home/yu/OneDrive/Construal')
model_dir = Path('/home/yu/OneDrive/Construal/pretrained-models/v3det')
os.chdir(model_dir)

# Initialize the DetInferencer
inferencer = DetInferencer(
    # DINO
    # model='/home/yu/OneDrive/Construal/pretrained-models/v3det/checkpoints/configs/v3det/dino-4scale_swin_16xb1_sample1e-3_v3det_36e.py', 
    # weights='/home/yu/OneDrive/Construal/pretrained-models/v3det/checkpoints/DINO_V3Det_SwinB.pth',

    # DETR (DETR is faster)
    model='/home/yu/OneDrive/Construal/pretrained-models/v3det/checkpoints/configs/v3det/deformable-detr-refine-twostage_swin_16xb2_sample1e-3_v3det_50e.py',
    weights='/home/yu/OneDrive/Construal/pretrained-models/v3det/checkpoints/Deformable_DETR_V3Det_SwinB.pth',

    device='cuda:1'
)

# image paths
img_path = Path('/home/yu/OneDrive/Construal/pretrained-models/v3det/data/V3Det/images')
img_paths = list(img_path.glob('**/*.jpg'))

for i, img_path in enumerate(tqdm(img_paths)):
    # get image name and folder
    img_name = img_path.stem
    img_folder = img_path.parent.stem
    
    # skip if image not exist
    if not img_path.exists():
        print(f'{img_path} does not exist')
        continue

    # inference
    # the model only keep the top 300 predictions (with the highest confidence)
    with io.capture_output() as captured:
        out = inferencer(str(img_path), show=False)['predictions'][0]

    # save results
    torch.save(out, wdir/f'data/v2/v3det/on-train/per-image-objects/{img_folder}-{img_name}.pt')
    