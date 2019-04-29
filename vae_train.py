import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from tensorboardX import SummaryWriter
from data_loader import *
from vae_nets import *
from util import metric_eval

seed = 8964
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

num_epochs = 60
batch_size = 8
restore = True
checkpoint_path = 'checkpoints/vae_checkpoint.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


# Define dataloaders
val_set = OccMapDataset('dataset/Cityscapes/CS_val_64.csv', transform=transforms.Compose([Rescale((256, 512)), ToTensor()]))
val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)
# Use train set for choosing hyper-parameters, and use train+val for final traning and testing
# train_set = OccMapDataset('dataset/Cityscapes/CS_train_64.csv', transform=transforms.Compose([Rescale((256, 512)), ToTensor()]))
train_set = OccMapDataset('dataset/Cityscapes/CS_trainplusval_64.csv', transform=transforms.Compose([Rescale((256, 512)), ToTensor()]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders = {'train': train_loader, 'val': val_loader}

model = vae_mapping()
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

if restore:
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path)
        epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    else:
        epoch = 0
else:
    epoch = 0


while epoch < num_epochs:
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        acc = 0.0
        iou = 0.0

        # Iterate over data.
        for i, temp_batch in enumerate(dataloaders[phase]):
            temp_rgb = temp_batch['rgb'].float().to(device)
            temp_map = temp_batch['map'].long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                pred_map, mu, logvar = model(temp_rgb, phase == 'train')
                loss, CE, KLD = loss_function_map(pred_map, temp_map, mu, logvar)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                else:
                    temp_acc, temp_iou = metric_eval(pred_map, temp_map)
                    acc += temp_acc
                    iou += temp_iou

            running_loss += loss.item()

            # tensorboardX logging
            if phase == 'train':
                writer.add_scalar(phase+'_loss', loss.item(), epoch * len(train_set) / batch_size + i)
                writer.add_scalar(phase+'_loss_CE', CE.item(), epoch * len(train_set) / batch_size + i)
                writer.add_scalar(phase+'_loss_KLD', KLD.item(), epoch * len(train_set) / batch_size + i)

            # statistics
        if phase == 'train':
            running_loss = running_loss / len(train_set)
            print(phase, running_loss)
        else:
            running_loss = running_loss / len(val_set)
            print(phase, running_loss, acc / len(val_set), iou / len(val_set))
            writer.add_scalar(phase+'_acc', acc.item()/len(val_set), (epoch + 1) * len(train_set) / batch_size)
            writer.add_scalar(phase+'_iou', iou.item()/len(val_set), (epoch + 1) * len(train_set) / batch_size)



    # save model per epoch
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        }, checkpoint_path)
    print('model after %d epoch saved...' % (epoch+1))
    epoch += 1

writer.close()
