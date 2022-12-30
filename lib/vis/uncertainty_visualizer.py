from PIL import Image as PILImage
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import functools
from torch.nn import functional as F
from torch import nn
import numpy as np
import cv2
import pdb
import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')


class UncertaintyVisualizer(object):
    def __init__(self, configer):
        super(UncertaintyVisualizer, self).__init__()
        
        self.configer = configer
        
    def vis_uncertainty(self, uncertainty):
        