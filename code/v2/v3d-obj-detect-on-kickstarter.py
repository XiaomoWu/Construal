# Using V3Det to detect objects in images, run on Kickstarter images
# This script is adapted from

import os
import torch

from pathlib import Path
from mmdet.apis import DetInferencer
from IPython.utils import io
from pyarrow.feather import write_feather, read_feather


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

# get pids to use
pjson = read_feather('/home/yu/OneDrive/Construal/data/pjson.feather')
valid_pids = pjson.loc[(pjson.category=='Accessories') | (pjson.category=='Product Design'), 'pid'].to_list()


# inference
img_root = Path('/home/yu/chaoyang/research-resources/kickstart-raw-from-amrita/kickstarter-image')
for i, pid in enumerate(valid_pids):
    # print progress
    if i % 100 == 0:
        print(f'Processing: {i}/{len(valid_pids)}')
    
    # construct image path
    img_path = img_root / f'{pid}/profile_full.jpg'
    if not img_path.exists():
        print(f'{img_path} does not exist')
        continue

    # inference
    # the model only keep the top 300 predictions (with the highest confidence)
    out = inferencer(str(img_path), show=False)['predictions'][0]

    # save results
    torch.save(out, wdir/f'data/v3det/on-kickstarter/per-image-objects/{pid}.pt')
    