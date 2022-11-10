# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Microsoft Research
# Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
# Copyright (c) 2019
# yuyua@microsoft.com
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.tools.logger import Logger as Log


SEG_MODEL_DICT = {
}


class ModelManager(object):
    """ 
    Check availability of the model defined in the configs, 
    and return the model if it exists.
    """

    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
