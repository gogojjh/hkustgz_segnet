from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils.tools.logger import Logger as Log
from utils.tools.average_meter import AverageMeter
from utils.vis.seg_visualizer import SegVisualizer
from loss.loss_manager import LossManager
from segmentor.tools.module_runner import ModuleRunner
from models.model_manager import ModelManager
from datasets.data_loader import DataLoader
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from utils.distributed import get_world_size, get_rank, is_distributed


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
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer)
        self.evaluator = get_evaluator(configer)
        
        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        
        self._init_model()
        
    def _init_model(self):
        """ 
        - Set up model and load the saved checkpoints, then put them to GPU devices.
        - Get network params associated with different learning rates, and set optimizer and secheduler.
        - Set dataloaders and losses.
        - Set the network based on the usage of contrastive learning or not.
        """
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)
        
        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()
            
        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)
        
        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss()
        # Put losses to GPU devices.
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)
        
        # ************* Use contrastive learning or not. ************* # 
        self.with_contrast = True if self.configer.exists('contrast') else False
        # ************* Use contrastive learning or not. ************* # 
        self.use_proto = True if self.configer.exists('prototype') else False
        
        Log.info('with_contrast: {}, use_proto: {}'.format(self.with_contrast, self.use_proto))
            
    def _get_parameters(self):
        """ 
        Different lrs for backbone and non-backbone params.
        """
        bb_lr = [] # backbone lr
        nbb_lr = [] # non-backbone lr
        paramas_dict = dict(self.seg_net.named_parameters())
        for key, value in paramas_dict.items():
            if 'backbone' not in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)
        
        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        
        return params
        
    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        
        return groups
                
    
        
    def __val(self, data_loader=None):
        """ 
        Validation during the training phase.
        """    
        self.seg_net.eval()
        self.pixel_loss.eval() 
        start_time = time.time()
        replicas = self.evaluator.prepare_validaton() # for DP 
        
        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                Log.info('{} images processed \n.'.format(j))
            # Get image data from the data dict loaded by dataloader, and send them to gpu
            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            # ************* Start Validation ************* # 
            with torch.no_grad():
                # diverse_size: send individual input to the net instead of in batch size
                if self.data_helper.conditions.diverse_size:
                    if is_distributed():
                        outputs = [self.seg_net(inputs[i]) for i in range(len(inputs))]
                    else:
                        outputs = nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
                    
                    for i in range(len(outputs)):
                        loss = self.pixel_loss(outputs[i], targets[i]).unsqueeze(0)
                        self.val_losses.update(loss.item(), 1)
                        outputs_i = outputs[i]['seg'] # todo
                        if isinstance(outputs_i, torch.Tensor):
                            outputs_i = [outputs_i]
                        self.evaluator.update_score(outputs, data_dict['meta'])
                
                else:
                    outputs = self.seg_net(*inputs, is_eval=True)
                    
                    try: 
                        loss = self.pixel_loss(outputs, targets)
                    except AssertionError as e:
                        print('Output Length: {}, Target Length:{}'.format(len(outputs), len(targets)))
                        
                    if not is_distributed():
                        outputs = self.module_runner.gather(outputs)
                    self.val_losses.update(loss.item(), batch_size)
                    
                    if isinstance(outputs, dict):
                        self.evaluator.update_score(outputs['seg'], data_dict['meta'])
                    else:
                        self.evaluator.update_score(outputs, data_dict['meta'])
            
            self.evaluator.update_performance()
            
            self.configer.update()
                    
                        
                    
                    
        
        
        
        
    
        
    def train(self):
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.
        
        