""" 
Dataset setup and loaders
"""
from datasets import hkustgz
import torchvision.transforms as standard_transforms


import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader


def setup_loaders(configs, args):
    """ 
    Set up data loader for HKUSTGZ dataset.
    """
    if configs['dataset'] == 'hkustgz':
        args.dataset_cls = hkustgz
        