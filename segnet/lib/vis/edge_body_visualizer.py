from lib.utils.distributed import get_rank, is_distributed
from lib.utils.tools.logger import Logger as Log
from lib.datasets.tools.transforms import DeNormalize

import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('Agg')


EDGE_DIR = 'vis/results/edge'
BODY_DIR = 'vis/results/body'


class EdgeBodyVisualizer(object):
    def __init__(self, configer):
        super(EdgeBodyVisualizer, self).__init__()

        self.configer = configer
        self.wandb_mode = self.configer.get('wandb', 'mode')

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']
            
    # def vis_body(self, )