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
from datetime import datetime
import logging


def make_exp_name(configs):
    """ 
    Create unique output dir name based on experiment configs
    """
    exp_name = '{}-{}'.format(configs['dataset']
                              ['name'], configs['model']['arch'])

    return exp_name


def save_log(prefix, output_dir, date_str, rank=0):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' +
                            date_str + '_rank_' + str(rank) + '.log')
    print('Logging: ', filename)
    logging.basicConfig(level=logging.INFO, format=fmt,
                        datefmt=date_fmt, filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    if rank == 0:
        logging.getLogger('').addHandler(console)
    else:
        fh = logging.FileHandler(filename)
        logging.getLogger('').addHandler(fh)


def prep_experiment(configs, args):
    """ 
    Create output directories, setup logging, initialize wandb.

    configs: from YAML file
    args: from argument parser
    """
    exp_name = make_exp_name(configs)

    # Initialize args for training recording.
    args.best_record = {'epoch': -1, 'iter': 0,
                        'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iou': 0}
    args. last_record = {}
    args.ngpu = torch.cuda.device_count()
    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    # Set up wandb.
    # Initialize wandb.
    wandb.init(project=exp_name, entity="hkustgz_segnet", config=argparse.Namespace(
        **configs), name=args.date_str, mode=configs['training']['wandb_mode'], settings=wandb.Settings(start_method='fork'))

    # Create dir for logging.
    args.exp_path = os.path.join(configs['training']['log_save_dir'], configs['dataset']
                                 ['name'] + '_' + configs['model']['arch'], exp_name)
    if args.local_rank == 0:
        os.makedirs(args.exp_path, exist_ok=True)
        save_log('log', args.exp_path, args.date_str, rank=args.local_rank)
        open(os.path.join(args.exp_path, args.date_str + '.txt'),
             'w').write(str(args) + '\n\n')
