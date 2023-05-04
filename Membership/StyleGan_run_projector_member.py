import torch
import pickle
import os
import sys
import scipy
from torchvision.utils import save_image
import dnnlib
import numpy as np
import pandas as pd
import json
import projector_z
import glob

MODEL_PTH= '/home/lauh/projects/def-gambsseb/lauh/latent_space/00000-cifar10_mini-cond-cifar/network-snapshot-004032.pkl'


with open('/scratch/lauh/cifar10_mini/dataset.json') as f:
    labels_dict = json.load(f)
  
label_df=pd.DataFrame(labels_dict['labels'], columns=['id', 'label'])

already_done=[]
for i in range(5):
    done_pth = glob.glob("/home/lauh/projects/def-gambsseb/lauh/latent_space/z_member/0000{}/*".format(i))
    for img_pth in done_pth:
        already_done.append('/'.join(img_pth.split('/')[-2:]))

print(already_done)

member_sample= label_df.loc[~label_df.id.isin(already_done)].sample(1000)
for _,row in member_sample.iterrows():
    projector_z.run_projection(MODEL_PTH, "/scratch/lauh/cifar10_mini/"+row['id'], row['label'],"/home/lauh/projects/def-gambsseb/lauh/latent_space/z_member/"+row['id'], False, 303, 1000)

