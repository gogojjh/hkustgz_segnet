import os

import cv2
import numpy as np

from lib.datasets.tools.transforms import DeNormalize
from lib.utils.tools.logger import Logger as Log

PROTOTYPE_DIR = 'vis/results/prototype'

# class PrototypeVisualier(object):
#     def __init__(self, configer):
#         super(PrototypeVisualier, self).__init__()
        
#         self.configer = configer
        
#         self.num_classes = self.configer.get('data', 'num_classes')
#         self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        
#     def vis_prototype(self, proto_mean, proto_var, img, contrast_target):
#         canvas = img.copy()
#         for i in range(self.num_classes):
#             for m in range(self.num_prototype):
                