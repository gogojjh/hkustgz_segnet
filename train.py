""" 
training code
User-defined params are set in YAML file, and other params are automatically set in the args.
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


if configs['training']['use_ddp']:
    torch.cuda.set_device(args.local_rank)
    print('Local rank: ', args.local_rank)
    # Initialize distributed communication.
    torch.distributed.init_process_group(backend='nccl', init_method='env://')


def main():
    """ 
    Load configs, set up wandb, dataloader, loss func, optimizer
    """

    prep_experiment(configs, args)
    train_loader, val_loader, train_set = datasets.setup_loaders(configs, args)
    criterion, criterion_val = loss.get_loss(configs, args)  # TODO
    net = network.get_net(configs, args, criterion)
    optim, scheduler = optimizer.get_optimizer(configs, args, net)

    # Wrap network into ddp.
    if configs['training']['use_ddp']:
        net = network.wrap_net_in_ddp(net, args)
    if configs['training']['snapshot']:
        optimizer.load_weights(net, optim, configs['training']['snapshot'],
                               configs['training']['restore_optimizer'])

    torch.cuda.empty_cache()

    # Main Loop
    for epoch in range(configs['training']['start_epoch'], configs['training']['max_epoch']):
        # Update epoch cfg
        # Prepare for updating the configs at the beginning of each epoch.
        cfg.immutable(False)
        cfg.EPOCH = epoch  # Update
        cfg.immutable(True)  # Set the configs to be immutable.

        scheduler.step()  # Update lr.

    def train(train_loader, net, optim, curr_epoch):
        """ 
        Runs the training loop per epoch.
        """
        net.train()
        
        


if __name__ == 'main':
    main()
