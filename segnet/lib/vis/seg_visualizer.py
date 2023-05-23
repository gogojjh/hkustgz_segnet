#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualizer for segmentation.


import os
import cv2
import numpy as np
import wandb

from lib.datasets.tools.transforms import DeNormalize
from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import get_rank, is_distributed
from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('Agg')

SEG_DIR = 'vis/results/seg'
ERROR_MAP_DIR = 'vis/results/error_map'

# num_classes = 25 + 1
FS_COLOR_MAP = np.array([
    [177, 165, 25], # 0: 'void/unlabelled'
    [0, 162, 170],  # 1: 'drivable road'
    [162, 248, 63],  # 2 'sidewalk'
    [241, 241, 241],  # 3:'parking'
    [147, 109, 6],  # 4 'curb'
    [122, 172, 9],  # 5 'bike Path'
    [12, 217, 219],  # 6 'road marking'
    [7, 223, 115],  # 7 'low-speed road'
    [161, 161, 149],  # 8 'lane'  
    [231, 130, 130],  # 9 'person'
    [252, 117, 35],  # 10 'rider'
    [92, 130, 216], # 11 'car'
    [0, 108, 114],  # 12 'bicycle'
    [0, 68, 213], # 13 'motorcycle'
    [91, 0, 231],  # 14 'truck'
    [227, 68, 255],  # 15 'building'
    [168, 38, 191],  # 16 fence
    [106, 0, 124],  # 17 'wall'
    [255, 215, 73],  # 18 'vegetation'
    [209, 183, 91],  # 19 'terrain'
    [244, 255, 152], # 'river'
    [138, 164, 165], # 'pole'
    [175, 0, 106], # 'traffic sign'
    [228, 0, 140], # 'traffic light'
    [234, 178, 200], # 'road block'
    [255, 172, 172] # 'sky'
])

# num_classes = 23 + 1
FS_CS_COLOR_MAP = np.array([
    [105, 105, 105], # 0: 'void/unlabelled'
    [128, 64, 128],  # 1: 'road'
    [92, 130, 216],  # 2 'bike path'
    [70, 70, 70],  # 3:'building'
    [102, 102, 156],  # 4 wall
    [190, 153, 153],  # 5 fence
    [153, 153, 153],  # 6 pole
    [250, 170, 30],  # 7 'traffic light'
    [220, 220, 0],  # 8 'traffic sign'
    [107, 142, 35],  # 9 'vegetation'
    [152, 251, 152],  # 10 'terrain'
    [70, 130, 180], # 11 sky
    [220, 20, 60],  # 12 person
    [255, 0, 0], # 13 rider
    [0, 0, 142],  # 14 car
    [0, 0, 70],  # 15 truck
    [0, 60, 100],  # 16 bus
    [0, 80, 100],  # 17 train
    [0, 0, 230],  # 18 'motorcycle'
    [147, 109, 6], # 19: 'bicycle'
    [119, 11, 32],  # 20 'curb'
    [244, 255, 152], # 21: 'river'
    [234, 178, 200], # 22: 'road block'
    [244, 35, 232],  # 23 'sidewalk'
])

# num_classes = 22 + 1
FS_CS_COLOR_MAP_V2 = np.array([
    [105, 105, 105], # 0: 'void/unlabelled'
    [128, 64, 128],  # 1: 'road'
    [244, 35, 232], # 2 'sidewalk'
    [70, 70, 70],  # 3:'building'
    [102, 102, 156],  # 4 wall
    [190, 153, 153],  # 5 fence
    [153, 153, 153],  # 6 pole
    [250, 170, 30],  # 7 'traffic light'
    [220, 220, 0],  # 8 'traffic sign'
    [107, 142, 35],  # 9 'vegetation'
    [152, 251, 152],  # 10 'terrain'
    [70, 130, 180], # 11 sky
    [220, 20, 60],  # 12 person
    [255, 0, 0], # 13 rider
    [0, 0, 142],  # 14 car
    [0, 0, 70],  # 15 truck
    [0, 60, 100],  # 16 bus
    [0, 80, 100],  # 17 train
    [0, 0, 230],  # 18 'motorcycle'
    [147, 109, 6], # 19: 'bicycle'
    [119, 11, 32],  # 20 'curb'
    [244, 255, 152], # 21: 'river'
    [234, 178, 200], # 22: 'road block'
])


# num_classes = 19 + 1
CS_COLOR_MAP = np.array([
    [105, 105, 105], # 0: 'void/unlabelled'
    [128, 64, 128],  # 1: 'road'
    [244, 35, 232],  # 2 'sidewalk'
    [70, 70, 70],  # 3:'building'
    [102, 102, 156],  # 4 wall
    [190, 153, 153],  # 5 fence
    [153, 153, 153],  # 6 pole
    [250, 170, 30],  # 7 'traffic light'
    [220, 220, 0],  # 8 'traffic sign'
    [107, 142, 35],  # 9 'vegetation'
    [152, 251, 152],  # 10 'terrain'
    [70, 130, 180], # 11 sky
    [220, 20, 60],  # 12 person
    [255, 0, 0], # 13 rider
    [0, 0, 142],  # 14 car
    [0, 0, 70],  # 15 truck
    [0, 60, 100],  # 16 bus
    [0, 80, 100],  # 17 train
    [0, 0, 230],  # 18 'motorcycle'
    [119, 11, 32],  # 19 'bicycle'
])


class SegVisualizer(object):

    def __init__(self, configer=None):
        self.configer = configer
        self.wandb_mode = self.configer.get('wandb', 'mode')

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']
            
    def __remap_pred(self, pred_img):
        ''' 
        Map the void/unlabelled class to 0, and class ids of other classes are increased by 1.
        ignore_label is -1
        '''
        
        return pred_img + 1

    def wandb_log_error_img(self, img_path, file_name):
        error_img = Image.open(img_path)

        im = wandb.Image(error_img, caption=file_name)

        if get_rank() == 0:
            wandb.log({'error image': [im]})
            
    def wandb_log_pred_img(self, img_path, file_name):
        pred_img = Image.open(img_path)

        im = wandb.Image(pred_img, caption=file_name)

        if get_rank() == 0:
            wandb.log({'pred image': [im]})
            
    def vis_pred(self, ori_img, pred, name='default'):
        base_dir = os.path.join(self.configer.get('train', 'out_dir'), SEG_DIR)
        c, h, w = ori_img.size()

        if not is_distributed() or get_rank() == 0:
            if not os.path.exists(base_dir):
                Log.info('Dir:{} not exists!'.format(base_dir))
                os.makedirs(base_dir)

        if not isinstance(pred, np.ndarray):
            if len(pred.size()) < 2:
                Log.error('Pred image size is not valid.')
                exit(1)
            if len(pred.size()) == 2:
                pred = pred.data.cpu().numpy()  # [h w]

        if pred.shape[0] != h or pred.shape[1] != w:
            pred = cv2.resize(
                pred, dsize=(w, h),
                interpolation=cv2.INTER_NEAREST)
        
        # ori image
        mean = self.configer.get('normalize', 'mean')
        std = self.configer.get('normalize', 'std')
        div_value = self.configer.get('normalize', 'div_value')
        ori_img = DeNormalize(div_value, mean, std)(ori_img)
        ori_img = ori_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [1024 2048 3]
        # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        # vis semantic image
        pred = self.__remap_pred(pred)
        pred_rgb_vis = pred
        if self.configer.get('data', 'num_classes') == 23:
            pred_rgb_vis = FS_CS_COLOR_MAP[pred].astype(np.uint8)
        elif self.configer.get('data', 'num_classes') == 25:
            pred_rgb_vis = FS_COLOR_MAP[pred].astype(np.uint8)
        elif self.configer.get('data', 'num_classes') == 22:
            pred_rgb_vis = FS_CS_COLOR_MAP_V2[pred].astype(np.uint8)
        else: 
            pred_rgb_vis = CS_COLOR_MAP[pred].astype(np.uint8)
       
        if self.configer.get('val', 'vis_pred') or self.configer.get('test', 'vis_pred'):
            weighted_img = cv2.addWeighted(ori_img, 0.5, pred_rgb_vis, 0.5, 0.0)

            #weighted_img = cv2.cvtColor(weighted_img, cv2.COLOR_RGB2BGR)           
            
            pred_path = os.path.join(base_dir, '{}_pred.png'.format(name))
            FileHelper.make_dirs(pred_path, is_file=True)
            ImageHelper.save(pred_rgb_vis, save_path=pred_path)
            Log.info('Saving {}_pred.png'.format(name))

            pred_path = os.path.join(base_dir, '{}_pred_orig.png'.format(name))
            FileHelper.make_dirs(pred_path, is_file=True)
            ImageHelper.save(weighted_img, save_path=pred_path)
            Log.info('Saving {}_pred_orig.png'.format(name))

            if self.wandb_mode == 'online':
                self.wandb_log_pred_img(pred_path, '{}_pred.jpg'.format(name))

        return pred, pred_rgb_vis

    def vis_error(self, pred, gt, name='default'):
        base_dir = os.path.join(self.configer.get('train', 'out_dir'), ERROR_MAP_DIR)

        if not is_distributed() or get_rank() == 0:
            if not os.path.exists(base_dir):
                Log.info('Dir:{} not exists!'.format(base_dir))
                os.makedirs(base_dir)

        if not isinstance(pred, np.ndarray):
            if len(pred.size()) < 2:
                Log.error('Pred image size is not valid.')
                exit(1)
            if len(pred.size()) == 2:
                pred = pred.data.cpu().numpy()  # [h w]

        if not isinstance(gt, np.ndarray):
            if len(gt.size()) < 2:
                Log.error('GT image size is not valid.')
                exit(1)
            if len(gt.size()) == 2:
                gt = gt.data.cpu().numpy()  # [h w]

        if pred.shape[0] != gt.shape[0] or pred.shape[1] != gt.shape[1]:
            pred = cv2.resize(
                pred, dsize=(gt.shape[1],
                             gt.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        
        # vis error map
        error_map = np.abs(gt - pred)
        error_map[gt == self.ignore_label] = 0  # ignore class
        error_map[error_map > 0] = 1

        fig = plt.figure()
        plt.axis('off')
        errormap = plt.imshow(error_map, cmap='viridis')
        # fig.colorbar(errormap)
        img_path = os.path.join(base_dir, '{}_error.png'.format(name))
        fig.savefig(img_path,
                    bbox_inches='tight', transparent=True, pad_inches=0.0)
        plt.close('all')
        Log.info('Saving {}_error.png'.format(name))

        if self.wandb_mode == 'online':
            self.wandb_log_error_img(img_path, '{}_error.jpg'.format(name))
