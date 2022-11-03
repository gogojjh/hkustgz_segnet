""" 
# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py

Get config params from args/yaml and pass them to the corresponding variables
"""

from __future__ import absolute_import     # import from internal packages
from __future__ import division     # accurate division operation
from __future__ import print_function
from __future__ import unicode_literals

import torch

from utils.attr_dict import AttrDict


# Attribute Dictionary 
__C = AttrDict()
cfg = __C
__C.EPOCH = 0

# Attribute Dictionary for Dataset
__C.DATASET = AttrDict()
# HKUSTGZ dataset dir location
__C.DATASET.HKUSTGZ_DIR = ''
# HKUSTGZ augmented dataset dir location
__C.DATASET.HKUSTGZ_AUG_DIR = ''
#Number of splits to support
__C.DATASET.CV_SPLITS = 3



