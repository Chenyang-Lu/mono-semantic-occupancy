import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import glob
from data_loader import *
from vae_nets import *
from util import vis_with_FOVmsk


root_dir = 'dataset/Cityscapes'
map_list = sorted(glob.glob(os.path.join(root_dir, 'Semantic_Occupancy_Grid_Multi_64', 'val', '*', '*occ_map.png')))
checkpoint_path = 'checkpoints/vae_checkpoint.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define dataloaders
test_set = OccMapDataset('dataset/Cityscapes/CS_test_64.csv', transform=transforms.Compose([Rescale((256, 512)), ToTensor()]))
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

model = vae_mapping()
model = model.to(device)

if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('trained model loaded...')
else:
    print('cannot load trained model...')
    exit()

model.eval()  # Set model to evaluate mode

# Iterate over data.
for i, temp_batch in enumerate(test_loader):
    print('example no. ', i)
    temp_rgb = temp_batch['rgb'].float().to(device)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        pred_map, mu, logvar = model(temp_rgb, False)

        map_to_save = np.reshape(np.argmax(pred_map.cpu().numpy().transpose((0, 2, 3, 1)), axis=3), [64, 64]).astype(np.uint8)
        io.imsave(map_list[i][:-4] + '_nn_pred.png', map_to_save)
        io.imsave(map_list[i][:-4] + '_nn_pred_c.png', vis_with_FOVmsk(map_to_save))
