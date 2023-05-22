# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: RainbowSecret, LayneH, Donny You
# Microsoft Research
# yuyua@microsoft.com
# Copyright (c) 2019
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import cv2

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from lib.vis.uncertainty_visualizer import UncertaintyVisualizer
from lib.vis.seg_visualizer import SegVisualizer


# Define colors (in RGB) for different labels
HKUSTGZ_COLOR_MAP = np.array([
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

FS_CS_COLOR_MAP = np.array([
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
    [147, 109, 6], # 19: 'curb
    [119, 11, 32],  # 20 'bicycle'
    [12, 217, 219], # 21 'road marking'
    [244, 255, 152], # 22: 'river'
    [234, 178, 200] # 23: 'road block'
])

class Tester(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.save_dir = self.configer.get('train', 'out_dir')
        self.seg_net = None
        self.test_loader = None
        self.test_size = None
        self.infer_time = 0
        self.infer_cnt = 0
        self._init_model()

        self.vis_prototype = self.configer.get('test', 'vis_prototype')
        self.vis_pred = self.configer.get('test', 'vis_pred')
        self.uncer_visualizer = UncertaintyVisualizer(configer=self.configer)
        self.seg_visualizer = SegVisualizer(configer)

        if self.vis_prototype:
            from lib.vis.prototype_visualizer import PrototypeVisualier
            self.proto_visualizer = PrototypeVisualier(configer=configer)

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        self.test_loader = None
        self.test_size = 1

        self.seg_net.eval()

    def get_ros_batch_data(self, data_dict):
        ''' 
        This function is called everytime when ros callback is called.
        '''
        self.test_loader = data_dict

    def __relabel(self, label_map):
        height, width = label_map.shape
        label_dst = np.zeros((height, width), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[label_map == i] = self.configer.get('data', 'label_list')[i]

        label_dst = np.array(label_dst, dtype=np.uint8)

        return label_dst
    
    def __remap_pred(self, pred_img):
        ''' 
        Map the void/unlabelled class to 0, and class ids of other classes are increased by 1.
        ignore_label is -1
        '''
        
        return pred_img + 1
        

    def test(self, ros_processor=None, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()

        Log.info('save dir {}'.format(self.save_dir))
        FileHelper.make_dirs(self.save_dir, is_file=False)

        sem_img_ros = []
        uncer_img_ros = []
        for j, data_dict in enumerate(self.test_loader):
            inputs = data_dict['img']
            names = data_dict['name']
            metas = data_dict['meta']

            with torch.no_grad():
                outputs = self.seg_net.forward(inputs) # [b c h w]
                logits = outputs['seg']  # [b c h w]
                ori_img_size = metas[0]['ori_img_size']
                uncertainty = outputs['confidence']  # [b h w k]
                uncertainty = F.interpolate(
                            input=uncertainty.unsqueeze(1), size=(ori_img_size[1], ori_img_size[0]),
                            mode='bilinear', align_corners=True)  # [b, 1, h, w]
                uncertainty = uncertainty.squeeze(1) # [b h w]
                batch_size = uncertainty.shape[0]
                batch_size = inputs.shape[0]
                for i in range(batch_size):
                    self.uncer_visualizer.vis_uncertainty(
                                        uncertainty[i], name='{}'.format(names[i]))
                    
                    # =========== vis prediction img =========== #
                    pred = torch.argmax(
                        logits, dim=1)  # [b h w]
                    pred_img, pred_rgb_vis = self.seg_visualizer.vis_pred(inputs[i], pred[i], names[i])
                    
                    if self.configer.get(
                            'ros', 'use_ros') and self.configer.get('phase') == 'test_ros':
                        ''' 
                        Publish semantic image rosmsg
                        Publish the uncertainty image:
                        - with top 3 largest uncertainties
                        - each channel: 1000 *(training class id(int) + prediciton confidence(0-1))  
                        '''
                        sem_img_ros.append(pred_img)
                        sem_img_ros.append(pred_rgb_vis)
                        
                        # =========== vis uncertainty img =========== #
                        # logits_i = cv2.resize(logits[i].cpu().numpy(),
                        #                     tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
        
                        logits = F.interpolate(
                            input=logits, size=(ori_img_size[1], ori_img_size[0]),
                            mode='bilinear', align_corners=True)  # [b, 1, 2048, 1536]
                        logits = logits[i].permute(1, 2, 0) # [2048, 1536, 23/num_cls]

                        # get the top 3 largest softmax class-wise prediction confidence
                        # logits_i = torch.from_numpy(logits_i).cuda()
                        m = nn.Softmax(dim=-1)
                        logits = m(logits)
                        val, ind = torch.topk(logits, k=3, dim=-1)  # [h, w, 3]
                        uncer_img = torch.zeros(
                            [ori_img_size[1],
                             ori_img_size[0],
                             3],
                            dtype=torch.int64)
                        for j in range(3):
                            ''' 
                            logits: [0, 1]
                            class id: int
                            confidence img: (class id[i] + logit[i]) * 100
                            '''
                            # [h, w, num_cls]([h, w])
                            uncer_img[:, :, j] = ((val[:, :, j] + ind[:, :, j]) * 100).long()

                        uncer_img = uncer_img.cpu().numpy()  # int64
                        uncer_img = uncer_img.astype(np.uint8)
                        # uncer_img = cv2.cvtColor(uncer_img, cv2.CV_16UC1)
                        uncer_img_ros.append(uncer_img)

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        # Print the log info & reset the states.
        Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))

        if self.configer.get('ros', 'use_ros') and self.configer.get('phase') == 'test_ros':
            return sem_img_ros, uncer_img_ros

    def ss_test(self, inputs, scale=1):
        outputs = self.seg_net.forward(inputs)
        torch.cuda.synchronize()

        if isinstance(outputs, list):
            outputs = outputs[-1]
        elif isinstance(outputs, dict):
            outputs = outputs['seg']
        elif isinstance(outputs, tuple):
            outputs = outputs[-1]
        outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
        return outputs

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]


if __name__ == "__main__":
    pass