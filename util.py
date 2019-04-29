import torch
import numpy as np
import py_img_seg_eval.eval_segm as eval_segm
import imageio
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def metric_eval(current_nn, current_gt):

    current_gt = current_gt.cpu().numpy().squeeze()
    current_nn = np.reshape(np.argmax(current_nn.cpu().numpy().transpose((0, 2, 3, 1)), axis=3), [64, 64])

    FOVmsk = imageio.imread('misc/mask_64.png')
    FOVmsk = FOVmsk[:, :, 0]
    valid_FOV_index = FOVmsk.reshape(-1) != 0

    valid_index = current_gt.reshape(-1) != 4
    valid_index = valid_index * valid_FOV_index

    current_gt = current_gt.reshape(-1)[valid_index]
    current_nn = current_nn.reshape(-1)[valid_index]

    current_gt = current_gt.reshape(1, -1)
    current_nn = current_nn.reshape(1, -1)

    # eval_segm.pixel_accuracy(current_nn, current_gt)
    acc = eval_segm.mean_accuracy(current_nn, current_gt)
    iou = eval_segm.mean_IU(current_nn, current_gt)
    # eval_segm.frequency_weighted_IU(current_nn, current_gt)
    return acc, iou


def vis_with_FOVmsk(curr_map):
    mask = imageio.imread('misc/mask_64.png')
    mask = mask[:, :, 0]
    valid_FOV_index = mask == 0

    color_list = np.array([[128, 64, 128],
                        [244, 35, 232],
                        [152, 251, 152],
                        [255, 0, 0],
                        [0, 0, 0]], dtype=np.uint8)
    curr_map[valid_FOV_index] = 4

    # print(curr_map)
    curr_map = np.repeat(np.repeat(curr_map, 8, axis=0), 8, axis=1).reshape(-1)
    curr_map_c = np.zeros((64*64*8*8, 3), dtype=np.uint8)
    for i in range(64*64*8*8):
        # print(curr_map[i])
        curr_map_c[i, :] = color_list[curr_map[i]]

    curr_map_c = np.reshape(curr_map_c, (64*8, 64*8, 3))

    return curr_map_c
