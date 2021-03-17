# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python-3.8
#     language: python
#     name: python3
# ---

# # Init

# +
# Init for script use
with open("/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py", 'r') as _:
    exec(_.read())

import datatable as dt
import glob
import mmcv
import torch
import shutil

from datatable import f
from tqdm.auto import tqdm

dt.init_styles()

WORK_DIR = '/home/yu/OneDrive/Construal'
MODEL_DIR = f'{WORK_DIR}/models/mmdetection'
DATA_DIR = f'{WORK_DIR}/data'

os.chdir(MODEL_DIR)
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
os.chdir(WORK_DIR)
# -

# # Label distribution

# +
LVIS_DATA_DIR = '/home/yu/Data/LVIS'

import json

with open(f'{LVIS_DATA_DIR}/lvis_v1_train.json') as ff:
    label_dist = dt.Frame(json.load(ff)['categories'])

label_dist.names = {'def': 'definition'}
label_dist = label_dist[:, 
      [f.id, f.name, f.definition, f.instance_count, f.image_count, f.frequency]]

sv('label_dist')
# -

labels

# + [markdown] toc-hr-collapsed=true
# # Obj Detect
# -

# ## config Model

# +
# Specify the path to model config and checkpoint file
config_file = f'{MODEL_DIR}/configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
checkpoint_file = f'{MODEL_DIR}/checkpoints/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth'

# build the model from a config file and a checkpoint file
model_cuda0 = init_detector(config_file, checkpoint_file, device='cuda:0')
model_cuda1 = init_detector(config_file, checkpoint_file, device='cuda:1')
# -

# ## move jpgs

# get pid of "product design"
ld('pjson', path=DATA_DIR)
pids = pjson[f.category=='Product Design',f.pid].to_list()[0]

pids[1]


# +
# Copy all jpgs in a project folder to '/Kickstarter Image/project_folder'
def copy_jpg(pid):
    pdir = f'{DATA_DIR}/Kickstarter Data/{pid}'
    for i, file in enumerate(os.listdir(pdir)):
        if file.endswith('.jpg'):
            file_full_path = f'{pdir}/{file}'
            target_dir = f'{DATA_DIR}/Kickstarter Image/{pid}'
            target_path = f'{target_dir}/{file}'

            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            shutil.copyfile(file_full_path, target_path)

for pid in tqdm(pids):
    copy_jpg(pid)


# +
# Move depth detection results to a new subfolder
# Only need to do onece
def move_depth_results():
    pids = os.listdir(f'{DATA_DIR}/Kickstarter Image/')

    for pid in tqdm(pids):
        pdir = f'{DATA_DIR}/Kickstarter Image/{pid}'
        for file in os.listdir(pdir):
            if file.endswith('md2.jpg') or file.endswith('md2.npy'):
                file_full_path = f'{pdir}/{file}'
                target_dir = f'{DATA_DIR}/Kickstarter Image/{pid}/depth results'
                target_path = f'{target_dir}/{file}'

                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)

                shutil.move(file_full_path, target_path)
                
# move_depth_results()


# -

# ## Inference

# ### Unit test

# +
target = f'{DATA_DIR}/Sharing'

def unit(pid, model):
    pdir = f'{DATA_DIR}/Kickstarter Image/{pid}'
    output = {}
    for i, file in enumerate(os.listdir(pdir)):
        if file.endswith('profile_full.jpg'):
            img = f'{pdir}/{file}'
            res = inference_detector(model, img)
            
            model.show_result(img, res, out_file=f'{target}/example_{i}.jpg')
            
for pid in tqdm(pids[:5]):
    unit(pid, model_cuda1)


# -

# ### Batch run

# +
# Start Detecting!
def obj_detect(pid, model):
    pdir = f'{DATA_DIR}/Kickstarter Image/{pid}'
    output = {}
    for i, file in enumerate(os.listdir(pdir)):
        if file.endswith('.jpg'):
            img = f'{pdir}/{file}'
            res = inference_detector(model, img)
            
            # get cat_idx and cat_n
            # res[0]: box
            # res[1]: segment
            box_res = {}
            for cat_idx, cat in enumerate(res[0]):
                if len(cat)>0:
                    box_res[cat_idx] = {'n':len(cat),
                                        'prob':cat[:,-1].tolist()}
            
            output[file] = box_res

    return output

x = obj_detect(pids[0], model_cuda0)


# +
# Start Detecting!
def obj_detect(pid, model):
    pdir = f'{DATA_DIR}/Kickstarter Image/{pid}'
    output = {}
    for i, file in enumerate(os.listdir(pdir)):
        if file.endswith('.jpg'):
            img = f'{pdir}/{file}'
            res = inference_detector(model, img)
            
            # get cat_idx and cat_n
            # res[0]: box
            # res[1]: segment
            box_res = {}
            for cat_idx, cat in enumerate(res[0]):
                if len(cat)>0:
                    box_res[cat_idx] = {'n':len(cat),
                                        'prob':cat[:,-1].tolist()}
            
            output[file] = box_res
            


    # save results
    save_dir = f'{pdir}/object results'
    save_path = f'{save_dir}/mrcnn_lvis.pt'
    if not os.path.exists(save_path):
        os.mkdir(save_dir)
    torch.save(output, save_path)
    
    
device_id = 0
print(f'Using GPU:{device_id}')

half = int(len(pids)/2)
pids = [pids[:half], pids[half:]]

model = init_detector(config_file, checkpoint_file, device=f'cuda:{device_id}')

for pid in tqdm(pids[device_id]):
    obj_detect(pid, model)
# -
# ## Retreving results for sharing
#

# get pid of "product design"
ld('pjson', path=DATA_DIR)
pids = pjson[f.category=='Product Design',f.pid].to_list()[0]


# +
target = f'{DATA_DIR}/Sharing/object detect results'
if not os.path.exists:
    os.mkdir(target)

def save_sharing(pid):
    pdir = f'{DATA_DIR}/Kickstarter Image/{pid}/object results'
    target_dir = f'{target}/{pid}'
    if not os.path.exists(target_dir):
        shutil.copytree(pdir, target_dir)

for pid in tqdm(pids):
    save_sharing(pid)
# +
import torch

root_dir = '/home/yu/OneDrive/Construal/data/Sharing/object detect results'

outputs = {}
for i, pid in enumerate(os.listdir(root_dir)):
    res = torch.load(f'{root_dir}/{pid}/mrcnn_lvis.pt')
    outputs[pid] = res
    
# outputs will be the collected results.
# It's a dictionary, the key is the project_id, for example, type
# outputs['843185506'], then the results for pid=843185506 will
# show up.

