import os

import cv2
import numpy as np
import wandb
import PIL

from lib.datasets.tools.transforms import DeNormalize
from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import get_world_size, get_rank, is_distributed

UNCERTAINTY_DIR = 'vis/results/uncertainty'


class UncertaintyVisualizer(object):
    ''' 
    Uncertainty refers to data uncertainty / predictive uncertainty.
    '''

    def __init__(self, configer):
        super(UncertaintyVisualizer, self).__init__()

        self.configer = configer
        self.wandb_mode = self.configer.get('wandb', 'mode')
    
    def wandb_log(self, uncer_img, file_name):
        im = wandb.Image(uncer_img, caption=file_name)
        wandb.log({'uncertainty image': [im]})
        
    def vis_uncertainty(self, uncertainty, name='default'):
        base_dir = os.path.join(self.configer.get('train', 'out_dir'), UNCERTAINTY_DIR)

        if not isinstance(uncertainty, np.ndarray):
            if len(uncertainty.size()) != 2:  # [b h w]
                Log.error('Tensor size of uncertainty is not valid.')
                exit(1)

            # uncertainty = uncertainty.data.cpu().numpy().transpose(1, 0)
            uncertainty = uncertainty.data.cpu().numpy()

        if not is_distributed() or get_rank() == 0:
            if not os.path.exists(base_dir):
                Log.error('Dir:{} not exists!'.format(base_dir))
                os.makedirs(base_dir)

        uncertainty_img = np.ones((uncertainty.shape[0], uncertainty.shape[1], 3))
        for i in range(uncertainty_img.shape[-1]):
            uncertainty_img[:, :, i] = uncertainty

        # uncertainty_img = cv2.resize(uncertainty, tuple(
        #     self.configer.get('val', 'data_transformer')['input_size']))
        uncertainty_img = cv2.normalize(uncertainty_img, None, 0, 255, cv2.NORM_MINMAX)
        uncertainty_img = uncertainty_img.astype(np.uint8)
        cv2.imwrite(os.path.join(base_dir, '{}_uncertainty.jpg'.format(name)), uncertainty_img)
        Log.info('Saving {}_uncertainty.jpg'.format(name))
        
        if self.wandb_mode == 'online':
            self.wandb_log(uncertainty_img, '{}_uncertainty.jpg'.format(name))
