# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: RainbowSecret, JingyiXie, LangHuang
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

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import wandb
import cv2
import numpy as np

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank, is_distributed
from lib.vis.uncertainty_visualizer import UncertaintyVisualizer
# from mmcv.cnn import get_model_complexity_info
from ptflops import get_model_complexity_info
from einops import rearrange, repeat


class Trainer(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        # self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.var_scheduler = None
        self.running_score = None

        self._init_model()

        self.vis_prototype = self.configer.get('val', 'vis_prototype')
        self.vis_pred = self.configer.get('val', 'vis_pred')
        if self.vis_prototype:
            from lib.vis.prototype_visualizer import PrototypeVisualier
            self.proto_visualizer = PrototypeVisualier(configer=configer)

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()

        try:
            flops, params = get_model_complexity_info(
                self.seg_net, (3, 512, 512))
            split_line = '=' * 30
            print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
                split_line, (3, 512, 512), flops, params))
            print('!!!Please be cautious if you use the results in papers. '
                  'You may need to check if all ops are supported and verify that the '
                  'flops computation is correct.')
        except:
            Log.info('Failed in getting model complexity info.')
            # pass

        self.seg_net = self.module_runner.load_net(self.seg_net)

        var_params_group = None
        Log.info('Params Group Method: {}'.format(
            self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            if self.configer.exists('var_lr', 'lr_policy'):
                params_group, var_params_group = self._get_parameters()
            else:
                params_group = self._get_parameters()

        self.optimizer, self.var_optimizer, self.scheduler, self.var_scheduler = self.optim_scheduler.init_optimizer(
            params_group, var_params_group)

        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss()
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)

        self.with_proto = True if self.configer.exists("protoseg") else False

        self.uncer_visualizer = UncertaintyVisualizer(configer=self.configer)

        self.use_boundary = self.configer.get('protoseg', 'use_boundary')

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

        assert len(list(module.parameters())) == len(
            group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(
            params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        fcn_lr = []
        var_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' in key:
                bb_lr.append(value)
            elif 'aux_layer' in key or 'upsample_proj' in key:
                fcn_lr.append(value)
            elif 'uncertainty_head' or 'boundary_head' or 'boundary_attention_module' in key:
                var_lr.append(value)
            else:
                nbb_lr.append(value)

        if self.configer.exists('var_lr', 'lr_policy'):
            params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                      {'params': fcn_lr, 'lr': self.configer.get(
                          'lr', 'base_lr') * 10},
                      {'params': nbb_lr, 'lr': self.configer.get(
                          'lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
            var_params = [
                {'params': var_lr, 'lr': self.configer.get('var_lr', 'base_lr')}]
            Log.info('base lr for uncertainty head: {}'.format(
                self.configer.get('var_lr', 'base_lr')))
            return params, var_params
        else:
            params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                      {'params': fcn_lr, 'lr': self.configer.get(
                          'lr', 'base_lr') * 10},
                      {'params': nbb_lr, 'lr': self.configer.get(
                          'lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
            return params

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()
        scaler = torch.cuda.amp.GradScaler()

        if "swa" in self.configer.get('lr', 'lr_policy'):
            normal_max_iters = int(self.configer.get(
                'solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get(
                'solver', 'max_iters') - normal_max_iters) // 5 + 1

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))

        for i, data_dict in enumerate(self.train_loader):
            self.configer.update(('phase',), 'train')
            self.optimizer.zero_grad()
            self.var_optimizer.zero_grad()
            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step(self.configer.get('iters'))
            else:
                self.scheduler.step(self.configer.get('epoch'))

            if self.configer.get('var_lr', 'metric') == 'iters':
                self.var_scheduler.step(self.configer.get('iters'))
            else:
                self.var_scheduler.step(self.configer.get('epoch'))

            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(
                    self.configer.get('iters'),
                    self.scheduler, self.optimizer, backbone_list=[0, ]
                )
            gt_boundary = None
            if self.use_boundary:
                (inputs, targets, gt_boundary), batch_size = self.data_helper.prepare_data(data_dict)
            else:
                (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()
            with torch.cuda.amp.autocast():
                if not self.with_proto:
                    outputs = self.seg_net(*inputs)
                else:
                    pretrain_prototype = True if self.configer.get(
                        'iters') < self. configer.get('protoseg', 'warmup_iters') else False
                    if gt_boundary is not None:
                        gt_boundary = gt_boundary[:, None, ...]
                    outputs = self.seg_net(
                        *inputs, gt_semantic_seg=targets[:, None, ...],
                        gt_boundary=gt_boundary,
                        pretrain_prototype=pretrain_prototype)

            self.foward_time.update(time.time() - foward_start_time)

            loss_start_time = time.time()
            if is_distributed():
                import torch.distributed as dist

                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.reduce(reduced_inp, dst=0)
                    return reduced_inp

                with torch.cuda.amp.autocast():
                    loss = self.pixel_loss(outputs, targets, gt_boundary=gt_boundary)
                    backward_loss = loss['loss']
                    seg_loss = reduce_tensor(
                        loss['seg_loss']) / get_world_size()
                    prob_ppc_loss = reduce_tensor(
                        loss['prob_ppc_loss']) / get_world_size()
                    prob_ppd_loss = reduce_tensor(
                        loss['prob_ppd_loss']) / get_world_size()
                    uncer_seg_loss = reduce_tensor(
                        loss['uncer_seg_loss']) / get_world_size()
                    display_loss = reduce_tensor(
                        backward_loss) / get_world_size()
                    
                    #todo debug
                    # backward_loss = loss
                    # seg_loss = reduce_tensor(
                    #     loss) / get_world_size()
                    # display_loss = reduce_tensor(
                    #     backward_loss) / get_world_size()
            else:
                # backward_loss = display_loss = self.pixel_loss(
                #     outputs, targets)
                loss_tuple = self.pixel_loss(
                    outputs, targets, gt_boundary=gt_boundary)

                backward_loss = display_loss = loss_tuple['loss']
                seg_loss = loss_tuple['seg_loss']
                prob_ppc_loss = loss_tuple['prob_ppc_loss']
                prob_ppd_loss = loss_tuple['prob_ppd_loss']

            self.train_losses.update(display_loss.item(), batch_size)
            self.loss_time.update(time.time() - loss_start_time)

            if get_rank() == 0:
                wandb.log({"Epoch": self.configer.get('epoch'),
                           "Train Iteration": self.configer.get('iters'),
                           "Loss": backward_loss,
                           "seg_loss": seg_loss,
                           "prob_ppc_loss": prob_ppc_loss,
                           "prob_ppd_loss": prob_ppd_loss})

            backward_start_time = time.time()

            # backward_loss.backward()
            # self.optimizer.step()
            scaler.scale(backward_loss).backward()
            scaler.step(self.optimizer)
            scaler.step(self.var_optimizer)
            scaler.update()

            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 and \
                    (not is_distributed() or get_rank() == 0):
                Log.info(
                    'Train Epoch: {0}\tTrain Iteration: {1}\t'
                    'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                    'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                    'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                    'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                    'Learning rate = {3}\tUncertainty Head Learning Rate = {4}\n'
                    'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'
                    'seg_loss={seg_loss:.5f} prob_ppc_loss={prob_ppc_loss:.5f} prob_ppd_loss={prob_ppd_loss:.5f} uncer_seg_loss={uncer_seg_loss:.5f}'.
                    format(
                        self.configer.get('epoch'),
                        self.configer.get('iters'),
                        self.configer.get('solver', 'display_iter'),
                        self.module_runner.get_lr(self.optimizer),
                        self.module_runner.get_lr(self.var_optimizer),
                        batch_time=self.batch_time, foward_time=self.foward_time,
                        backward_time=self.backward_time, loss_time=self.loss_time,
                        data_time=self.data_time, loss=self.train_losses, seg_loss=seg_loss,
                        prob_ppc_loss=prob_ppc_loss, prob_ppd_loss=prob_ppd_loss,
                        uncer_seg_loss=uncer_seg_loss))

                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # save checkpoints for swa
            if 'swa' in self.configer.get('lr', 'lr_policy') and \
                    self.configer.get('iters') > normal_max_iters and \
                    ((self.configer.get('iters') - normal_max_iters) % swa_step_max_iters == 0 or
                     self.configer.get('iters') == self.configer.get('solver', 'max_iters')):
                self.optimizer.update_swa()

            if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            # if self.configer.get('epoch') % self.configer.get('solver', 'test_interval') == 0:
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                print('---------------------start validation---------------------')
                self.__val()

            del data_dict, inputs, targets, outputs

        self.configer.plus_one('epoch')

    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.configer.update(('phase',), 'val')

        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        replicas = self.evaluator.prepare_validaton()

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                if is_distributed():
                    dist.barrier()  # Synchronize all processes
                Log.info('{} images processed\n'.format(j))

            gt_boundary = None

            if self.configer.get('dataset') == 'lip':
                (inputs, targets, inputs_rev, targets_rev), batch_size = self.data_helper.prepare_data(
                    data_dict, want_reverse=True)
            elif self.configer.get('uncertainty_visualizer', 'vis_uncertainty'):
                (inputs, targets, names,
                 imgs), batch_size = self.data_helper.prepare_data(data_dict)
            elif self.configer.get('protoseg', 'use_boundary'):
                (inputs, targets, gt_boundary), batch_size = self.data_helper.prepare_data(data_dict)
            else:
                (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            with torch.no_grad():
                if self.configer.get('uncertainty_visualizer', 'vis_uncertainty'):
                    pretrain_prototype = True if self.configer.get(
                        'iters') < self. configer.get('protoseg', 'warmup_iters') else False
                    outputs = self.seg_net(*inputs, gt_semantic_seg=targets[:, None, ...],
                                            pretrain_prototype=pretrain_prototype)
                else:
                    outputs = self.seg_net(*inputs)

                if not is_distributed():
                    outputs = self.module_runner.gather(outputs)
                if isinstance(outputs, dict):
                    # ============== visualize==============#
                    if self.configer.get('uncertainty_visualizer', 'vis_uncertainty'):
                        # [b h w] [1, 256, 512]
                        # uncertainty = outputs['uncertainty']
                        h, w = targets.size(1), targets.size(2)
                        uncertainty = outputs['confidence']  # [b h w k]
                        # uncertainty = uncertainty.mean(-1)  # [b h w]
                        uncertainty = F.interpolate(
                            input=uncertainty.unsqueeze(1), size=(h, w),
                            mode='bilinear', align_corners=True)  # [b, 1, h, w]
                        uncertainty = uncertainty.squeeze(1)
                        if (j % (self.configer.get(
                                'uncertainty_visualizer', 'vis_inter_iter'))) == 0:
                            vis_interval_img = self.configer.get(
                                'uncertainty_visualizer', 'vis_interval_img')
                            batch_size = uncertainty.shape[0]
                            if vis_interval_img <= batch_size:
                                for i in range(0, vis_interval_img, batch_size):
                                    self.uncer_visualizer.vis_uncertainty(
                                        uncertainty[i], name='{}'.format(names[i]))
                                    inputs = data_dict['img']
                                    pred = outputs['seg']  # [b c h w]
                                    pred = torch.argmax(
                                        pred, dim=1)  # [b h w]
                                    
                                    pred_img, pred_rgb_vis = self.seg_visualizer.vis_pred(inputs[i], pred[i], names[i])
                                    # self.seg_visualizer.vis_error(
                                    #     pred[i], targets[i], names[i])
                                    
                                    self.seg_visualizer
                    if self.vis_prototype and self.configer.get('iters') % (self.configer.get('solver', 'test_interval') * 4) == 0:
                        if (j % (self.configer.get(
                                'uncertainty_visualizer', 'vis_inter_iter'))) == 0:
                            inputs = data_dict['img']
                            # metas = data_dict['meta']
                            names = data_dict['name']
                            sim_mat = outputs['logits']  # [(b h w) (c m)]
                            pred = outputs['seg']  # [b c h w]
                            num_classes = self.configer.get('data', 'num_classes')
                            num_prototype = self.configer.get('protoseg', 'num_prototype')
                            b, _, h, w, = pred.size()
                            sim_mat = sim_mat.reshape(b, h, w, num_classes, num_prototype)  # [b h w c m]
                            n = pred.shape[0]  # b
                            for k in range(n):
                                # ori_img_size = metas[k]['ori_img_size']  # [2048, 1024]
                                # border_size = metas[k]['border_size']  # [2048, 1024]
                                # inputs[k]: [3, 1024, 2048]
                                self.proto_visualizer.vis_prototype(
                                    sim_mat[k], inputs[k], names[k])

                    outputs = outputs['seg']
                self.evaluator.update_score(outputs, data_dict['meta'])

                del outputs

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.evaluator.update_performance()

        self.module_runner.save_net(self.seg_net, save_mode='performance')
        cudnn.benchmark = True

        # Print the log info & reset the states.
        self.evaluator.reduce_scores()
        if not is_distributed() or get_rank() == 0:
            self.evaluator.print_scores()

        self.batch_time.reset()
        self.evaluator.reset()
        self.seg_net.train()
        self.pixel_loss.train()

    def train(self):
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.__val(
                    data_loader=self.data_loader.get_valloader(dataset='val'))
                # return
            elif self.configer.get('network', 'resume_train'):
                self.__val(
                    data_loader=self.data_loader.get_valloader(dataset='train'))
                # return

        # return

        # if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
        #     self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
        #     return

        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
            self.__train()

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))

    def summary(self):
        from lib.utils.summary import get_model_summary
        import torch.nn.functional as F
        self.seg_net.eval()

        for j, data_dict in enumerate(self.train_loader):
            print(get_model_summary(self.seg_net, data_dict['img'][0:1]))
            return


if __name__ == "__main__":
    pass
