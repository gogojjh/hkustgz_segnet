from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.tools.average_meter import AverageMeter
from utils.vis.seg_visualizer import SegVisualizer
from loss.loss_manager import LossManager
from segmentator.tools.module_runner import ModuleRunner
from models.model_manager import ModelManager


class SDCTrainer(object):
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
