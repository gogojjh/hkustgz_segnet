import os

import cv2
import numpy as np
import torch
from PIL import Image

from lib.datasets.tools.transforms import DeNormalize
from lib.utils.tools.logger import Logger as Log
from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
from einops import rearrange, repeat

PROTOTYPE_DIR = '/vis/results/prototype'


def get_prototoype_colors():
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    num_cls = 20
    colors = [0] * (num_cls * 3)
    colors[0:3] = (0, 0, 0)  # 0: 'road'
    colors[3:6] = (244, 35, 232)  # 1 'sidewalk'
    colors[6:9] = (70, 70, 70)  # 2''building'
    colors[9:12] = (102, 102, 156)  # 3 wall
    colors[12:15] = (190, 153, 153)  # 4 fence
    colors[15:18] = (153, 153, 153)  # 5 pole
    colors[18:21] = (250, 170, 30)  # 6 'traffic light'
    colors[21:24] = (220, 220, 0)  # 7 'traffic sign'
    colors[24:27] = (107, 142, 35)  # 8 'vegetation'
    colors[27:30] = (152, 251, 152)  # 9 'terrain'
    colors[30:33] = (70, 130, 180)  # 10 sky
    colors[33:36] = (220, 20, 60)  # 11 person
    colors[36:39] = (255, 0, 0)  # 12 rider
    colors[39:42] = (0, 0, 142)  # 13 car
    colors[42:45] = (0, 0, 70)  # 14 truck
    colors[45:48] = (0, 60, 100)  # 15 bus
    colors[48:51] = (0, 80, 100)  # 16 train
    colors[51:54] = (0, 0, 230)  # 17 'motorcycle'
    colors[54:57] = (119, 11, 32)  # 18 'bicycle'
    colors[57:60] = (105, 105, 105)
    return colors


class PrototypeVisualier(object):
    def __init__(self, configer):
        super(PrototypeVisualier, self).__init__()

        self.configer = configer

        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.save_dir = self.configer.get('test', 'out_dir')
        self.ignore_label = -1
        if self.configer.exists(
                'loss', 'params') and 'ce_ignore_index' in self.configer.get(
                'loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def vis_prototype(self, sim_mat, ori_img, name):
        '''
        sim_mat: [h w c m]
        ori_img: inputs[k] [3 h w]
        '''
        # ori image
        mean = self.configer.get('normalize', 'mean')
        std = self.configer.get('normalize', 'std')
        div_value = self.configer.get('normalize', 'div_value')
        ori_img = DeNormalize(div_value, mean, std)(ori_img)
        ori_img = ori_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [1024 2048 3]
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        h, w = sim_mat.shape[0], sim_mat.shape[1]

        sem_pred = torch.max(sim_mat, dim=-1)[0] # [h w c]
        sem_pred = torch.argmax(sem_pred, dim=-1) # [h w]
        sim_mat = rearrange(sim_mat, 'h w c m -> h w (c m)') # [h w (c m)]
        proto_pred = torch.argmax(sim_mat, dim=-1) # [h w]
        proto_pred = proto_pred % self.num_prototype # proto id inside the predicted cls
        # save an img for each class
        for i in range(1, self.num_classes):
            mask = sem_pred == i
            
            
            proto_pred = Image.fromarray(proto_pred, 'P')
            colors = get_prototoype_colors()
            proto_pred.putpalette(colors)
            proto_pred = np.asarray(proto_pred.convert('RGB'), np.uint8)

            proto_pred = cv2.addWeighted(ori_img, 0.5, proto_pred, 0.5, 0.0)

            save_path = os.path.join(PROTOTYPE_DIR,
                                     '{}_{}_cls_proto.png'.format(name, i))
            FileHelper.make_dirs(save_path, is_file=True)
            ImageHelper.save(proto_pred, save_path=save_path)
            Log.info('{} saved.'.format(save_path))
