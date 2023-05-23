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


UNCERTAINTY_DIR = 'vis/results/uncertainty'


class UncertaintyVisualizer(object):
    '''
    Uncertainty refers to data uncertainty / predictive uncertainty.
    '''

    def __init__(self, configer):
        super(UncertaintyVisualizer, self).__init__()

        self.configer = configer
        self.wandb_mode = self.configer.get('wandb', 'mode')

        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def wandb_log(self, img_path, file_name):
        uncer_img = Image.open(img_path)

        im = wandb.Image(uncer_img, caption=file_name)
        if get_rank() == 0:
            wandb.log({'uncertainty image': [im]})

    def vis_uncertainty(self, uncertainty, name='default'):
        base_dir = os.path.join(self.configer.get('train', 'out_dir'), UNCERTAINTY_DIR)

        if not isinstance(uncertainty, np.ndarray):
            if len(uncertainty.size()) != 2:  # [b h w]
                Log.error('Tensor size of uncertainty is not valid.')
                exit(1)

            uncertainty = uncertainty.data.cpu().numpy()

        if not is_distributed() or get_rank() == 0:
            if not os.path.exists(base_dir):
                Log.error('Dir:{} not exists!'.format(base_dir))
                os.makedirs(base_dir)

        fig = plt.figure()
        plt.axis('off')
        heatmap = plt.imshow(uncertainty, cmap='viridis')
        # fig.colorbar(heatmap)
        img_path = os.path.join(base_dir, '{}_uncertainty.png'.format(name))
        fig.savefig(img_path,
                    bbox_inches='tight', transparent=True, pad_inches=0.0)
        plt.close('all')
        Log.info('Saving {}_uncertainty.jpg'.format(name))

        if self.wandb_mode == 'online':
            self.wandb_log(img_path, '{}_uncertainty.jpg'.format(name))
