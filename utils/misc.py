""" 
Miscellaneous Functions
"""

import sys
import os
import torch
import wandb
import argparse
from ruamel.yaml import YAML
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils


def make_exp_name(configs):
    """ 
    Create unique output dir name based on experiment configs
    """
    exp_name = '{}-{}'.format(configs['dataset']
                              ['name'], configs['model']['arch'])

    return exp_name


def prep_experiment(configs, args):
    """ 
    Create output directories, setup logging, initialize wandb.

    configs: from YAML file
    args: from argument parser
    """
    exp_name = make_exp_name(configs)

    # Set up wandb
    # Initialize wandb.
    wandb.init(project=exp_name, entity="hkustgz_segnet", config=argparse.Namespace(
        **configs), name=configs['training']['trial_name'], mode=configs['training']['wandb_mode'], settings=wandb.Settings(start_method='fork'))

    # initialize args for training recording
    args.best_record = {'epoch': -1, 'iter': 0,
                        'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iou': 0}
    args. last_record = {}

    # create dir for logging
    args.exp_path = os.path.join(configs['training']['log_save_dir'], configs['dataset']
                                 ['name']+'_'+configs['model']['arch'], exp_name)
    if args.local_rank == 0:
        os.makedirs(args.exp_path, exist_ok=True)
