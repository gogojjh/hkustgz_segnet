# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
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

from utils.distributed import is_distributed

from utils.tools.logger import Logger as Log


SEG_LOSS_DICT = {}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _paralle(self, loss):
        if is_distributed():  # use DDP
            Log.info('Use distributed loss.')
            return loss

        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1: # use DP
            Log.info('use DataParallelCriterion loss')
            from extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get(
            'loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!').format(key)
            exit(1)
        Log.info('Loss: {}'.format(key))
        loss = SEG_LOSS_DICT[key](self.configer)
        return self._paralle(loss)
