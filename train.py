""" 
training code
"""
from ruamel.yaml import YAML
import wandb
import argparse
import logging
import os
import torch
from apex import amp  # mixed-float training for speed-up

from config import cfg
from utils.misc import prep_experiment
import datasets
import loss
import network
import optimizer


# Argument Parser
parser = argparse.ArgumentParser(
    description='HKUSTGZ Semantic Segmentation')
# ----------------------- Please change the config file name ----------------------- #
parser.add_argument('-c', '--config_name', type=str,
                    default='hkustgz.yaml', help='config file name')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
args = parser.parse_args()

# Yaml Loader
root_name = os.path.dirname(os.path.abspath(__file__))
configs = YAML().load(open(root_name+'/configs/' + args.config_name))


# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1

# If test mode is true, run two epochs
if configs['training']['test_mode']:
    configs['training']['max_epoch'] = 2


if configs['training']['apex']:
    torch.cuda.set_device(args.local_rank)
    print('Local rank: ', args.local_rank)
    # Initialize distributed communication.
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


def main():
    """ 
    Load configs, set up wandb, dataloader, loss func, optimizer
    """

    prep_experiment(configs, args)
    


if __name__ == 'main':
    main()
