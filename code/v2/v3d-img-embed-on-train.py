# Use the backbone of V3D (a SWIM Transformer) to generate image embeddings
# of kickstarter images

import os
import torch
import mmcv


from pathlib import Path
from mmdet.apis import DetInferencer
from tqdm import tqdm
from IPython.utils import io
from pyarrow.feather import write_feather, read_feather

# set working directory
wdir = Path('/home/yu/OneDrive/Construal')

# select device
device = 'cuda:0'

# change the working directory to the model directory
# otherwise the model will not be loaded correctly
model_dir = Path('/home/yu/OneDrive/Construal/pretrained-models/v3det')
os.chdir(model_dir)

# Initialize the DetInferencer
inferencer = DetInferencer(
    # DETR (DETR is faster)
    model='/home/yu/OneDrive/Construal/pretrained-models/v3det/checkpoints/configs/v3det/deformable-detr-refine-twostage_swin_16xb2_sample1e-3_v3det_50e.py',
    weights='/home/yu/OneDrive/Construal/pretrained-models/v3det/checkpoints/Deformable_DETR_V3Det_SwinB.pth',

    device=device
)

# modify the model to remove the last laters (only keep the backbone)
model = inferencer.model.backbone
model.eval()

# image paths
img_path = Path('/home/yu/OneDrive/Construal/pretrained-models/v3det/data/V3Det/images')
img_paths = sorted(img_path.glob('**/*.jpg'))

# split img_paths into two roughtly equal parts
midpoint = len(img_paths) // 2
img_paths_1 = img_paths[:midpoint]
img_paths_2 = img_paths[midpoint:]

# determine the image paths to use depending on the device
if device == 'cuda:1':
    img_paths = img_paths_1
elif device == 'cuda:0':
    img_paths = img_paths_2

# compute embeddings
for i, img_path in enumerate(tqdm(img_paths)):

    # skip if img_path not exists
    if not img_path.exists():
        print(f'{img_path} does not exist')
        continue

    # construct save path
    save_path = wdir/f'data/v2/v3det/on-train/per-image-embed/{img_path.stem}.pt'
    
    # skip if save_path already exists
    if save_path.exists():
        continue

    # get feature map
    for out in inferencer.preprocess([mmcv.imread(img_path)]):

        # get the last feature map
        feature_map = model(out[1]['inputs'][0].float().unsqueeze(0).to(device))[-1]

        # average pooling
        # unlike CNN, Transformer models don't pool. So we need to pool manually
        feature_map = torch.nn.AdaptiveAvgPool2d((1, 1))(feature_map)

        # flatten
        feature_map = feature_map.flatten().cpu().detach().numpy()

    # save results
    torch.save(feature_map, save_path)
    