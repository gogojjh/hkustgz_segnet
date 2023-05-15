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
import timeit
import cv2
import collections

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
from lib.metrics.running_score import RunningScore
from lib.vis.seg_visualizer import SegVisualizer
from lib.vis.palette import get_cityscapes_colors, get_ade_colors, get_lip_colors, get_camvid_colors
from lib.vis.palette import get_pascal_context_colors, get_cocostuff_colors, get_pascal_voc_colors, get_autonue21_colors
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from scipy import ndimage
from PIL import Image
from math import ceil

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
        self.seg_data_loader = DataLoader(configer)
        self.save_dir = self.configer.get('test', 'out_dir')
        self.seg_net = None
        self.test_loader = None
        self.test_size = None
        self.infer_time = 0
        self.infer_cnt = 0
        self._init_model()

        self.vis_prototype = self.configer.get('test', 'vis_prototype')
        self.vis_pred = self.configer.get('test', 'vis_pred')

        if self.vis_prototype:
            from lib.vis.prototype_visualizer import PrototypeVisualier
            self.proto_visualizer = PrototypeVisualier(configer=configer)

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        '''if 'test' in self.save_dir:
            self.test_loader = self.seg_data_loader.get_testloader()
            self.test_size = len(self.test_loader) * self.configer.get('test', 'batch_size')
        else:
            self.test_loader = None
            self.test_size = 1'''
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
        image_id = 0

        Log.info('save dir {}'.format(self.save_dir))
        FileHelper.make_dirs(self.save_dir, is_file=False)

        save_prob = False
        if self.configer.get('test', 'save_prob') or self.configer.get('ros', 'use_ros'):
            save_prob = self.configer.get('test', 'save_prob')

            def softmax(X, axis=0):
                max_prob = np.max(X, axis=axis, keepdims=True)
                X -= max_prob
                X = np.exp(X)
                sum_prob = np.sum(X, axis=axis, keepdims=True)
                X /= sum_prob
                return X

        sem_img_ros = []
        uncer_img_ros = []
        for j, data_dict in enumerate(self.test_loader):
            inputs = data_dict['img']
            names = data_dict['name']
            metas = data_dict['meta']
            if 'subfolder' in data_dict:
                subfolder = data_dict['subfolder']

            if '/val/' in self.save_dir:  # and os.environ.get('save_gt_label'):
                labels = data_dict['labelmap']

            with torch.no_grad():
                # Forward pass.
                if self.configer.get('test', 'mode') == 'ss_test':
                    outputs = self.ss_test(inputs)

                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
                    n = outputs.shape[0]
                else:
                    outputs = [output.permute(0, 2, 3, 1).cpu().numpy().squeeze()
                               for output in outputs]
                    n = len(outputs)

                for k in range(n):
                    image_id += 1
                    ori_img_size = metas[k]['ori_img_size']
                    border_size = metas[k]['border_size']
                    logits = cv2.resize(outputs[k][:border_size[1], :border_size[0]],
                                        tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)

                    # save the logits map
                    if self.configer.get('test', 'save_prob'):
                        prob_path = os.path.join(self.save_dir, "prob/", '{}.npy'.format(names[k]))
                        FileHelper.make_dirs(prob_path, is_file=True)
                        np.save(prob_path, softmax(logits, axis=-1))
                    #! semantic prediction
                    label_img = np.asarray(np.argmax(logits, axis=-1), dtype=np.uint8)

                    #! use gpu for faster speed
                    # get the top 3 largest softmax class-wise prediction confidence
                    if self.configer.get(
                            'ros', 'use_ros') and self.configer.get('phase') == 'test_ros':
                        logits = torch.from_numpy(logits).cuda()
                        m = nn.Softmax(dim=-1)
                        logits = m(logits)
                        val, ind = torch.topk(logits, k=3, dim=-1)  # [h, w, 3]
                        uncer_img = torch.zeros(
                            [label_img.shape[0],
                             label_img.shape[1],
                             3],
                            dtype=torch.int64)
                        for i in range(3):
                            ''' 
                            logits: [0, 1]
                            class id: int
                            confidence img: (class id[i] + logit[i]) * 100
                            '''
                            # [h, w, num_cls]([h, w])
                            uncer_img[:, :, i] = ((val[:, :, i] + ind[:, :, i]) * 100).long()

                        uncer_img = uncer_img.cpu().numpy()  # int64
                        uncer_img = uncer_img.astype(np.uint8)
                        # uncer_img = cv2.cvtColor(uncer_img, cv2.CV_16UC1)
                        uncer_img_ros.append(uncer_img)
                        
                    label_img = self.__remap_pred(label_img)
                    if self.configer.exists('dataset_train') and len(self.configer.get('dataset_train')) > 1: 
                        color_img_ = FS_CS_COLOR_MAP[label_img].astype(np.uint8)
                    else: 
                        if self.configer.get('dataset') == 'hkustgz':
                            color_img_ = HKUSTGZ_COLOR_MAP[color_img_].astype(np.uint8)
                        elif self.configer.get('dataset') == 'cityscapes':
                            color_img_ = CS_COLOR_MAP[color_img_].astype(np.uint8)
                        else: 
                            raise RuntimeError('Invalid dataset type.')
                    # save semantic image
                    vis_path = os.path.join(self.save_dir, "vis/", '{}.png'.format(names[k]))
                    FileHelper.make_dirs(vis_path, is_file=True)
                    ImageHelper.save(color_img_, save_path=vis_path)

                    if self.configer.get(
                            'ros', 'use_ros') and self.configer.get('phase') == 'test_ros':
                        ''' 
                        Publish semantic image rosmsg
                        Publish the uncertainty image:
                        - with top 3 largest uncertainties
                        - each channel: 1000 *(training class id(int) + prediciton confidence(0-1))  
                        '''
                        sem_img_ros.append(label_img)

                    if self.vis_pred:
                        # =============== visualie ===============　#
                        from lib.datasets.tools.transforms import DeNormalize
                        mean = self.configer.get('normalize', 'mean')
                        std = self.configer.get('normalize', 'std')
                        div_value = self.configer.get('normalize', 'div_value')
                        org_img = DeNormalize(div_value, mean, std)(inputs[k])
                        org_img = org_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

                        color_img_ = cv2.resize(
                            color_img_, (org_img.shape[1],
                                        org_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

                        # save semantic overlay image
                        sys_img_part = cv2.addWeighted(org_img, 0.5, color_img_, 0.5, 0.0)

                        sys_img_part = cv2.cvtColor(sys_img_part, cv2.COLOR_RGB2BGR)

                        for i in range(0, 200):
                            mask = np.zeros_like(color_img_)
                            mask[color_img_ == i] = 1

                            contours = cv2.findContours(
                                mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
                            cv2.drawContours(sys_img_part, contours, -1, (255, 255, 255),
                                                1, cv2.LINE_AA)

                        vis_overlay_path = os.path.join(self.save_dir, "vis_overlay/",
                                                '{}.png'.format(names[k]))
                        FileHelper.make_dirs(vis_overlay_path, is_file=True)
                        ImageHelper.save(sys_img_part, save_path=vis_overlay_path)

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        # Print the log info & reset the states.
        Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))

        if self.configer.get('ros', 'use_ros') and self.configer.get('phase') == 'test_ros':
            return sem_img_ros, uncer_img_ros

    def ss_test(self, inputs, scale=1):
        if isinstance(inputs, torch.Tensor):
            n, c, h, w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
            scaled_inputs = F.interpolate(
                inputs, size=(int(h * scale),
                              int(w * scale)),
                mode="bilinear", align_corners=True)
            start = timeit.default_timer()
            outputs = self.seg_net.forward(scaled_inputs)
            torch.cuda.synchronize()
            end = timeit.default_timer()

            if isinstance(outputs, list):
                outputs = outputs[-1]
            elif isinstance(outputs, dict):
                outputs = outputs['seg']
            elif isinstance(outputs, tuple):
                outputs = outputs[-1]
            outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
            return outputs
        elif isinstance(inputs, collections.Sequence):
            device_ids = self.configer.get('gpu')
            replicas = nn.parallel.replicate(self.seg_net.module, device_ids)
            scaled_inputs, ori_size, outputs = [], [], []
            for i, d in zip(inputs, device_ids):
                h, w = i.size(1), i.size(2)
                ori_size.append((h, w))
                i = F.interpolate(
                    i.unsqueeze(0),
                    size=(int(h * scale),
                          int(w * scale)),
                    mode="bilinear", align_corners=True)
                scaled_inputs.append(i.cuda(d, non_blocking=True))
            scaled_outputs = nn.parallel.parallel_apply(
                replicas[:len(scaled_inputs)], scaled_inputs)
            for i, output in enumerate(scaled_outputs):
                outputs.append(
                    F.interpolate(
                        output[-1],
                        size=ori_size[i],
                        mode='bilinear', align_corners=True))
            return outputs
        else:
            raise RuntimeError("Unsupport data type: {}".format(type(inputs)))

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]


if __name__ == "__main__":
    pass
