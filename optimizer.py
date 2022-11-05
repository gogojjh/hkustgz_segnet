""" 
Set up optimizer and scheduler, and related weight loading functions.
"""

import logging
import torch
from torch import optim
import math


def get_optimizer(configs, args, net):
    """ 
    Choose optimizer (Adam or SGD).
    """
    param_groups = net.parameters()

    if configs['training']['optimizer']['name'] == 'sgd':
        optimizer = optim.SGD(param_groups,
                              lr=configs['training']['optimizer']['lr'],
                              weight_decay=configs['training']['optimizer']['weight_decay'],
                              momentum=configs['training']['optimizer']['momentum'],
                              nesterov=False)
    elif configs['training']['optimizer']['name'] == 'adam':
        amsgrad = False
        if configs['training']['optimizer']['amsgrad']:
            amsgrad = True
        optimizer = optim.Adam(param_groups,
                               lr=configs['training']['optimizer']['lr'],
                               weight_decay=configs['training']['optimizer']['weight_decay'],
                               amsgrad=amsgrad)
    else:
        raise ValueError('Not a valid optimizer.')

    if configs['training']['scheduler']['name'] == 'poly':
        def lambda1(epoch): return math.pow(
            1 - epoch / configs['training']['max_epoch'], configs['training']['scheduler']['poly_exp'])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('Unknown lr scheduler {}.'.format(
            configs['training']['scheduler']['name']))

    return optimizer, scheduler


def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """ 
    Load weights from snapshot file.
    """
    logging.info('Loading weights from model %s', snapshot_file)


def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """ 
    Restore weights and optimizer (optional) for resuming the work.
    """
    checkpoint = torch.load(
        snapshot, map_location=torch.device('cpu'))  # load the saved obj
    logging.info('Checkpoint loading completed.')
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = restore_forgiving_state(net, checkpoint['state_dict'])
    else:
        net = restore_forgiving_state(net, checkpoint)

    return net, optimizer


def restore_forgiving_state(net, loaded_dict):
    """ 
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained over a different number of classes. 
    So when the size of the current network doesn't equal to the loaded parameters, we skip these parameters.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            logging.info('Skip loading parameter %s', k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)

    return net
