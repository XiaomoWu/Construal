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

# change the working directory to the model directory
# otherwise the model will not be loaded correctly
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

# modify the model to remove the last laters (only keep the backbone)
model = inferencer.model.backbone
model.eval()

# get pids to use
pjson = read_feather('/home/yu/OneDrive/Construal/data/v1/pjson.feather')
valid_pids = pjson.loc[(pjson.category=='Accessories') | (pjson.category=='Product Design'), 'pid'].to_list()

# compute embeddings
img_root = Path('/home/yu/chaoyang/research-resources/kickstart-raw-from-amrita/kickstarter-image')
for i, pid in enumerate(tqdm(valid_pids)):
    
    # construct image path
    img_path = img_root / f'{pid}/profile_full.jpg'
    if not img_path.exists():
        print(f'{img_path} does not exist')
        continue

    # get feature map
    for out in inferencer.preprocess([mmcv.imread(img_path)]):

        # get the last feature map
        feature_map = model(out[1]['inputs'][0].float().unsqueeze(0).to('cuda:1'))[-1]

        # average pooling
        # unlike CNN, Transformer models don't pool. So we need to pool manually
        feature_map = torch.nn.AdaptiveAvgPool2d((1, 1))(feature_map)

        # flatten
        feature_map = feature_map.flatten().cpu().detach().numpy()

    # save results
    torch.save(feature_map, wdir/f'data/v2/v3det/on-kickstarter/per-image-embed/{pid}.pt')
    